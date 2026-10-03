"""实验会话：配置 + 增量协方差 + 结果版本，全部落盘、可重启续写。

并发模型：
* 每个会话一把 ``threading.RLock``，同一会话的追加/估计串行化，
  因此“两批同时到达”的最终状态严格等价于按某顺序串行追加，
  不丢批、不重复计入；
* 批次号在锁内检查并落盘，重复提交整批跳过（幂等）；
* 不同会话用不同的锁，互不阻塞。
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

import numpy as np

from ..doa.engine import (
    ArraySpec,
    EstConfig,
    estimate,
    validate_array_inputs,
)
from ..doa.beamform import beamscan_powers
from ..errors import (
    LimitExceededError,
    SessionNotFoundError,
    TrackNotFoundError,
    ValidationError,
)
from .accumulator import CovarianceAccumulator
from .events import EventBroker
from .store import SessionStore
from ..tracker import Tracker


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Session:
    def __init__(self, session_id: str, config: dict, store: SessionStore,
                 broker: EventBroker | None = None):
        self.id = session_id
        self.config = config
        self._store = store
        self._broker = broker
        self.lock = threading.RLock()
        errs = config["array"].get("position_errors")
        self._spec = ArraySpec(
            elements=int(config["array"]["elements"]),
            spacing=float(config["array"]["spacing"]),
            wavelength=float(config["array"]["wavelength"]),
            position_errors=tuple(errs) if errs else None,
        )
        upd = config.get("update") or {}
        self._acc = CovarianceAccumulator(
            m=self._spec.elements,
            mode=upd.get("mode", "sliding"),
            window_size=upd.get("window_size", 500),
            forgetting_factor=upd.get("forgetting_factor"),
        )
        trk = config.get("tracking") or {}
        self._tracking_enabled = bool(trk.get("enabled", False))
        self._tracker = Tracker(enabled=self._tracking_enabled)
        self._seen_batches: set[str] = set()
        self._version = 0
        self._latest_result: dict | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        state = self._store.load_state(self.id)
        buf = self._store.load_buffer_npz(self.id)
        acc_state = state.get("accumulator")
        if acc_state is not None:
            self._acc = CovarianceAccumulator.from_state(
                acc_state,
                buffer_real=buf["real"] if buf else None,
                buffer_imag=buf["imag"] if buf else None,
            )
        self._seen_batches = self._store.load_batch_ids(self.id)
        self._version = int(state.get("version", 0))
        self._latest_result = state.get("latest_result")
        if self._latest_result is None and state.get("latest_result_ref"):
            # 新版 state 只存信封骨架，完整信封以 results.jsonl 为准
            rows = self._store.load_results(self.id)
            ref_v = state["latest_result_ref"]["version"]
            for row in reversed(rows):
                if row["version"] == ref_v:
                    self._latest_result = row
                    break
        tracker_state = state.get("tracker")
        if tracker_state is not None:
            self._tracker.load_state(tracker_state)
        self._loaded = True

    # ------------------------------------------------------------------
    @property
    def spec(self) -> ArraySpec:
        return self._spec

    def est_config(self, overrides: dict | None = None) -> EstConfig:
        """会话默认配置 + 本次请求临时覆盖（仅覆盖显式给出的字段）。

        注意布尔字段（smoothing）不能用真值判断：显式传 False 必须覆盖。
        """
        cfg = dict(self.config.get("estimation") or {})
        if overrides:
            cfg.update({
                k: v for k, v in overrides.items()
                if v is not None and k in EstConfig.__dataclass_fields__
            })
        return EstConfig(
            angle_min=float(cfg.get("angle_min", -90.0)),
            angle_max=float(cfg.get("angle_max", 90.0)),
            angle_step=float(cfg.get("angle_step", 1.0)),
            smoothing=bool(cfg.get("smoothing", False)),
            subarray_length=cfg.get("subarray_length"),
            source_count=cfg.get("source_count"),
        )

    def status(self) -> dict:
        with self.lock:
            self._ensure_loaded()
            return {
                "session_id": self.id,
                "config": self.config,
                "version": self._version,
                "total_appended_snapshots": self._acc.total_appended,
                "active_snapshots": self._acc.n_snapshots,
                "n_accepted_batches": len(self._seen_batches),
                "latest_result_version": self._latest_result["version"]
                if self._latest_result
                else None,
                "created_from": self.config.get("created_at"),
                "tracking_enabled": self._tracking_enabled,
                "n_tracks": len(self._tracker.list_tracks())
                if self._tracking_enabled
                else 0,
            }

    # ------------------------------------------------------------------
    def append(
        self,
        batch_id: str,
        real: np.ndarray,
        imag: np.ndarray,
        *,
        run_estimate: bool = False,
        overrides: dict | None = None,
    ) -> dict:
        with self.lock:
            self._ensure_loaded()
            duplicate = batch_id in self._seen_batches
            if not duplicate:
                data = validate_array_inputs(
                    np.asarray(real, dtype=np.float64),
                    np.asarray(imag, dtype=np.float64),
                )
                self._acc.add_batch(data)
                self._seen_batches.add(batch_id)
                self._persist_after_append(batch_id)
            response = {
                "session_id": self.id,
                "batch_id": batch_id,
                "duplicate": duplicate,
                "active_snapshots": self._acc.n_snapshots,
                "total_appended_snapshots": self._acc.total_appended,
                "version": self._version,
            }
            if run_estimate and not duplicate:
                result = self._run_estimate(overrides, trigger_batch_id=batch_id)
                response["result"] = result
            return response

    def estimate(self, overrides: dict | None = None) -> dict:
        with self.lock:
            self._ensure_loaded()
            return self._run_estimate(overrides)

    # ------------------------------------------------------------------
    def _run_estimate(
        self, overrides: dict | None, trigger_batch_id: str | None = None
    ) -> dict:
        true_angles = overrides.get("true_angles") if overrides else None
        result = estimate(
            data=None,
            spec=self._spec,
            cfg=self.est_config(overrides),
            covariance=self._acc.covariance(),
            n_snapshots=self._acc.n_snapshots,
            effective_snapshots=self._acc.effective_snapshots,
            true_angles=true_angles,
        )
        self._version += 1
        envelope = {
            "version": self._version,
            "session_id": self.id,
            "created_at": _utc_now(),
            "trigger_batch_id": trigger_batch_id,
            "result": result,
        }
        if self._tracking_enabled:
            envelope["tracking"] = self._update_tracks(self._version, result)
        self._store.append_results(self.id, [envelope])
        self._save_state()
        self._latest_result = envelope
        if self._broker is not None:
            self._broker.publish(self.id, envelope)
        return envelope

    def _update_tracks(self, version: int, result: dict) -> dict:
        """用本次估计的角度推进航迹；事件补时间戳后追加落盘。"""
        powers = beamscan_powers(
            result["angles"],
            self._acc.covariance(),
            self._spec.positions(),
            self._spec.wavelength,
        )
        tracking = self._tracker.update(version, result["angles"], powers)
        events = tracking["events"]
        for ev in events:
            ev["created_at"] = _utc_now()
        if events:
            self._store.append_track_events(self.id, events)
        return tracking

    def _save_state(self) -> None:
        # state.json 里的 latest_result 只存信封骨架（结果体可能很大：
        # MUSIC 谱、准则曲线；航迹快照也只留当前状态）。重启加载时用
        # results.jsonl 最后一行恢复完整信封，历史逐拍信封永不回改。
        latest = self._latest_result
        latest_for_state = None
        if latest is not None:
            latest_for_state = {
                "version": latest["version"],
                "session_id": latest["session_id"],
                "created_at": latest.get("created_at"),
                "trigger_batch_id": latest.get("trigger_batch_id"),
            }
        state = {
            "version": self._version,
            "latest_result_ref": latest_for_state,
            "accumulator": self._acc.to_state(),
        }
        if self._tracking_enabled:
            state["tracker"] = self._tracker.to_state()
        self._store.save_state(self.id, state)

    def _persist_after_append(self, batch_id: str) -> None:
        self._store.record_batch_ids(self.id, [batch_id])
        packed = self._acc.pack_buffer()
        if packed is not None:
            self._store.save_buffer_npz(self.id, packed)
        self._save_state()

    # ------------------------------------------------------------------
    def results_since(self, version: int) -> list[dict]:
        with self.lock:
            self._ensure_loaded()
            all_results = self._store.load_results(self.id)
            return [r for r in all_results if r["version"] > int(version)]

    def latest_result(self) -> dict | None:
        with self.lock:
            self._ensure_loaded()
            return self._latest_result

    # ------------------------------------------------------------------
    # 航迹查询
    # ------------------------------------------------------------------
    def _tracks_or_404(self) -> Tracker:
        if not self._tracking_enabled:
            raise ValidationError(
                "该会话创建时未开启航迹（tracking.enabled=false），"
                "没有航迹可查"
            )
        return self._tracker

    def tracks_overview(self, include_terminated: bool = True) -> dict:
        with self.lock:
            self._ensure_loaded()
            tracker = self._tracks_or_404()
            return {
                "session_id": self.id,
                "tracks": [
                    t.snapshot()
                    for t in tracker.list_tracks(
                        include_terminated=include_terminated
                    )
                ],
            }

    def track_detail(self, track_id: int) -> dict:
        with self.lock:
            self._ensure_loaded()
            tracker = self._tracks_or_404()
            try:
                track = tracker.get_track(int(track_id))
            except KeyError:
                raise TrackNotFoundError(
                    f"航迹 {track_id} 不存在",
                    details={"track_id": int(track_id)},
                )
            snap = track.snapshot()
            snap["session_id"] = self.id
            snap["history"] = track.history
            return snap

    def track_events_since(self, version: int) -> list[dict]:
        with self.lock:
            self._ensure_loaded()
            self._tracks_or_404()
            rows = self._store.load_track_events(self.id)
            return [e for e in rows if int(e["version"]) > int(version)]


class SessionManager:
    def __init__(self, data_dir: str, broker: EventBroker | None = None):
        self.store = SessionStore(data_dir)
        self.broker = broker or EventBroker()
        self._sessions: dict[str, Session] = {}
        self._registry_lock = threading.Lock()

    def _validate_config(self, config: dict) -> dict:
        arr = config.get("array")
        if not arr or "elements" not in arr:
            raise ValidationError("缺少 array.elements 阵列配置")
        m = int(arr["elements"])
        if m < 2:
            raise ValidationError("阵元数至少为 2")
        if m > 64:
            raise LimitExceededError(
                f"阵元数 {m} 超过上限 64", details={"value": m, "max": 64}
            )
        wl = float(arr.get("wavelength", 0))
        sp = float(arr.get("spacing", 0))
        if wl <= 0:
            raise ValidationError(f"载波波长必须为正数，收到 {wl!r}")
        if sp <= 0:
            raise ValidationError(f"阵元间距必须为正数，收到 {sp!r}")
        errs = arr.get("position_errors")
        if errs is not None:
            errs = list(errs)
            if len(errs) != m:
                raise ValidationError(
                    f"position_errors 长度 {len(errs)} 必须等于阵元数 {m}"
                )
            arr["position_errors"] = errs
        est = config.get("estimation") or {}
        if est:
            amin = float(est.get("angle_min", -90.0))
            amax = float(est.get("angle_max", 90.0))
            step = float(est.get("angle_step", 1.0))
            if amax <= amin:
                raise ValidationError("angle_max 必须大于 angle_min")
            if step <= 0:
                raise ValidationError("angle_step 必须为正")
            n_grid = int((amax - amin) / step) + 1
            if n_grid > 200_000:
                raise LimitExceededError(
                    f"扫描点数 {n_grid} 超过上限 200000",
                    details={"value": n_grid, "max": 200000},
                )
        upd = config.get("update") or {}
        mode = upd.get("mode", "sliding")
        if mode not in ("sliding", "forgetting"):
            raise ValidationError(f"update.mode 仅支持 sliding/forgetting，收到 {mode!r}")
        if mode == "sliding":
            ws = int(upd.get("window_size", 500))
            if ws <= 0:
                raise ValidationError("window_size 必须为正整数")
            upd["window_size"] = ws
        else:
            ff = upd.get("forgetting_factor", 0.98)
            ff = float(ff)
            if not (0.0 < ff <= 1.0):
                raise ValidationError("forgetting_factor 必须满足 0 < lambda <= 1")
            upd["forgetting_factor"] = ff
        config["update"] = upd
        trk = config.get("tracking") or {}
        if not isinstance(trk, dict):
            raise ValidationError("tracking 必须是对象，含布尔字段 enabled")
        config["tracking"] = {"enabled": bool(trk.get("enabled", False))}
        return config

    def create_session(self, config: dict, session_id: str | None = None) -> Session:
        config = self._validate_config(dict(config))
        config["created_at"] = _utc_now()
        sid = session_id or uuid.uuid4().hex
        with self._registry_lock:
            if self.store.exists(sid):
                raise ValidationError(f"会话 {sid} 已存在")
            self.store.create(sid, config)
            session = Session(sid, config, self.store, self.broker)
            self._sessions[sid] = session
            return session

    def get(self, session_id: str) -> Session:
        with self._registry_lock:
            session = self._sessions.get(session_id)
            if session is not None:
                return session
            if not self.store.exists(session_id):
                raise SessionNotFoundError(f"会话不存在：{session_id}")
            config = self.store.load_config(session_id)
            session = Session(session_id, config, self.store, self.broker)
            session._ensure_loaded()
            self._sessions[session_id] = session
            return session

    def list_sessions(self) -> list[dict]:
        ids = set(self.store.list_sessions()) | set(self._sessions.keys())
        out = []
        for sid in sorted(ids):
            try:
                out.append(self.get(sid).status())
            except SessionNotFoundError:
                continue
        return out

    def publish_result(self, session_id: str, envelope: dict) -> None:
        self.broker.publish(session_id, envelope)

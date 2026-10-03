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
from ..errors import (
    LimitExceededError,
    SessionNotFoundError,
    ValidationError,
)
from .accumulator import CovarianceAccumulator
from .events import EventBroker
from .store import SessionStore
from .tracker import AngleTracker, Detection, Track, TrackEvent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _tracking_config(config: dict) -> dict | None:
    """从会话配置取航迹参数；未开启或老会话（无该字段）返回 None。

    开启但未覆盖任何门限时返回空 dict（表示用跟踪模块默认参数），
    与“未开启”的 None 区分开。
    """
    tc = config.get("tracking")
    if not tc:
        return None
    if not tc.get("enabled", False):
        return None
    return {k: v for k, v in tc.items() if k != "enabled" and v is not None}


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
        self._seen_batches: set[str] = set()
        self._version = 0
        self._latest_result: dict | None = None
        self._loaded = False
        # 航迹跟踪仅在会话显式开启时存在；不开的会话行为与旧版完全一致。
        self._tracking_overrides = _tracking_config(config)
        self._tracker: AngleTracker | None = (
            AngleTracker(self._tracking_overrides)
            if self._tracking_overrides is not None
            else None
        )

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
        if self._tracker is not None:
            tracker_state = state.get("tracker")
            if tracker_state is not None:
                self._tracker = AngleTracker.from_state(tracker_state)
            # 正常情况下 tracks.jsonl 是 state.json 的派生缓存；这里只做
            # 防御性补齐（state 有、事件日志缺行不会发生，因为同锁内先写
            # 日志再写 state）。
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
                "tracking_enabled": self._tracker is not None,
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
        track_events: list[TrackEvent] = []
        if self._tracker is not None:
            tracking = self._update_tracking(self._version, result)
            track_events = tracking["events"]
            # 信封要整体 JSON 落盘，事件换成普通 dict
            tracking["events"] = [e.to_dict() for e in track_events]
            envelope["tracking"] = tracking
        self._store.append_results(self.id, [envelope])
        if track_events:
            self._store.append_track_events(
                self.id,
                [{
                    "version": self._version,
                    "session_id": self.id,
                    "created_at": envelope["created_at"],
                    "events": [e.to_dict() for e in track_events],
                }],
            )
        self._save_state()
        self._latest_result = envelope
        if self._broker is not None:
            self._broker.publish(self.id, envelope)
        return envelope

    def _update_tracking(self, version: int, result: dict) -> dict:
        """用本次 MUSIC 结果推进航迹，返回挂到估计信封上的 tracking 块。

        角度取 ``result["angles"]``（已按角度升序），功率特征取对应细化
        角上的 MUSIC 谱高（相对量，只在同一次估计内做对比，足够用于
        护航判据与未来扩展）。归属结论与 angles 等长、一一对应。
        """
        assert self._tracker is not None
        angles = list(result["angles"])
        spectrum = result.get("spectrum_at_angles") or []
        detections = [
            Detection(
                angle=float(a),
                log_power=float(
                    np.log(max(float(spectrum[i]), np.finfo(float).tiny))
                ) if i < len(spectrum) else 0.0,
                detection_index=i,
            )
            for i, a in enumerate(angles)
        ]
        upd = self._tracker.update(version, detections)
        by_index = {a.detection_index: a for a in upd.assignments}
        angle_tracks = [
            {
                "angle": angles[i],
                "track_id": by_index[i].track_id,
                "associated": by_index[i].associated,
                "reason": by_index[i].reason,
            }
            for i in range(len(angles))
        ]
        return {
            "version": version,
            "angle_tracks": angle_tracks,
            "events": upd.events,
            "tracks": [t.to_dict() for t in self._tracker.list_tracks()],
        }

    def _save_state(self) -> None:
        state = {
            "version": self._version,
            "latest_result": self._latest_result,
            "accumulator": self._acc.to_state(),
        }
        if self._tracker is not None:
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
    # 航迹查询（仅开启航迹的会话有内容）
    # ------------------------------------------------------------------
    @property
    def tracking_enabled(self) -> bool:
        return self._tracker is not None

    def _require_tracking(self) -> AngleTracker:
        if self._tracker is None:
            raise ValidationError("该会话未开启航迹跟踪")
        return self._tracker

    def list_tracks(self, include_terminated: bool = True) -> list[dict]:
        with self.lock:
            self._ensure_loaded()
            tracker = self._require_tracking()
            return [self._track_payload(t) for t in
                    tracker.list_tracks(include_terminated=include_terminated)]

    def get_track_history(self, track_id: int) -> dict:
        with self.lock:
            self._ensure_loaded()
            tracker = self._require_tracking()
            track = tracker.get_track(int(track_id))
            if track is None:
                raise SessionNotFoundError(f"航迹不存在：{track_id}")
            return self._track_payload(track, with_history=True)

    def track_events_since(self, version: int) -> list[dict]:
        """带版本号轮询航迹事件：只返回严格新于 ``version`` 的版本批次。

        每个估计版本至多一行（行内是该次产生的若干事件，可能为空——
        空事件版本不落盘），因此不漏、不重。
        """
        with self.lock:
            self._ensure_loaded()
            self._require_tracking()
            rows = self._store.load_track_events(self.id)
            return [r for r in rows if int(r["version"]) > int(version)]

    def _track_payload(self, track: Track, *, with_history: bool = False) -> dict:
        payload = track.to_dict()
        if with_history:
            payload["history"] = [
                {"version": pt.version, "angle": pt.angle, "kind": pt.kind}
                for pt in track.history
            ]
        return payload


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
        tracking = config.get("tracking")
        if tracking is not None:
            tracking = dict(tracking)
            enabled = bool(tracking.get("enabled", False))
            tracking["enabled"] = enabled
            int_fields = (
                "confirm_hits", "tentative_max_misses", "lost_after_misses",
                "terminate_after_misses", "convoy_max_age", "grace_scans",
            )
            float_fields = (
                "tentative_gate_deg", "confirm_gate_deg", "position_alpha",
                "convoy_enter_deg",
                "convoy_region_deg", "convoy_margin_deg", "convoy_wide_deg",
                "convoy_release_deg",
            )
            for name in int_fields:
                if tracking.get(name) is not None:
                    tracking[name] = int(tracking[name])
            for name in float_fields:
                if tracking.get(name) is not None:
                    tracking[name] = float(tracking[name])
            config["tracking"] = tracking
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

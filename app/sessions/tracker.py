"""角度航迹跟踪（不依赖 HTTP / 会话存储）。

每次 MUSIC 估计吐出一组按角度排序的峰，本模块把这些峰逐次串成带稳定
编号的航迹，并给出航迹生命周期事件：

    tentative（新建/暂时）→ confirmed（确认）
    confirmed → lost（连续若干次未被看到）→ confirmed（重新看到）
    tentative/lost → terminated（更久未被看到，编号永久封存）

普通关联用“全局最近邻 + 波门”：航迹按匀速模型外推一步，最近的唯一
峰关联到航迹；波门内没有峰计一次漏检。

难点在交叉。两个相向目标从彼此身边穿过时，交叉附近 MUSIC 谱常只剩
一个峰、MDL 少报一个源，还会冒出远离交叉中心的假峰（本阵列下典型是
±28° 旁瓣假峰）。这时既不能把某条航迹判丢失，也不能把两条并成一条，
分开后编号还得各自跟着运动方向走、不能互换。本实现采用

    **运动趋势护航（convoy coasting）**

* 当两条*已确认*航迹相向（速度异号）且一步外推位置进入 ``convoy_enter``
  度以内，编成一个护航组：记录交叉中心与各自的运动方向（符号），进入
  “接近（approach）”相位。
* approach 相位：两条航迹都不做峰关联，只按各自速度线性外推（coast），
  计历史点但标 ``coast``、*不计漏检*；本次检测里剩余的峰不产生新航迹，
  也不归属任何航迹（标 ``crossing``）。这样单峰、少峰、假峰都伤害不到
  航迹，不会误判丢失、不会合并。
* 恢复（两峰重新分开）判据对线性外推漂移免疫：要求区域内重新出现
  *分居交叉中心两侧* 的两个峰，且两航迹预测都已越过中心
  ``convoy_margin`` 度；用两峰中点对预测整体平移（消除共同漂移）后，
  各自残差不超过 ``convoy_wide``。满足后按**进入护航时冻结的运动方向**
  认领峰——与角度排序无关，所以交叉之后编号不互换。进入“认回
  （assigned）”相位。
* assigned 相位继续按冻结方向把本侧峰认给航迹（这一段峰间距仍可能
  小于常规波门，普通 GNN 会串号），直到两峰分开到 ``convoy_release``
  度以上才解除护航，交回普通 GNN。

为什么选它、放弃了什么、代价、以及在什么条件下会认错，见
``docs/航迹跟踪设计.md``。

幂等/确定性：本模块只做纯函数式的状态推进，不接触时间、随机数或 IO；
给定同样的检测序列必然得到同样的航迹与事件，因此重启续写、并发按
实际顺序串行回放的结果逐项一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

TrackState = Literal["tentative", "confirmed", "lost", "terminated"]
PointKind = Literal["measurement", "coast"]
EventType = Literal[
    "track_created",
    "track_confirmed",
    "track_lost",
    "track_reacquired",
    "track_terminated",
]
AssignReason = Literal["associated", "unassigned", "crossing"]

# 默认门限/生命周期参数（度、次）。按“每批 50 快拍、每批一次估计”的
# 验收节奏标定，可在建会话时整体覆盖。
DEFAULT_PARAMS: dict[str, float] = {
    "tentative_gate_deg": 5.0,    # 暂态航迹关联波门
    "confirm_gate_deg": 10.0,     # 确认航迹关联波门（含 lost 重捕）
    "confirm_hits": 3,            # 连续命中达到此次数才确认
    "tentative_max_misses": 2,    # 暂态航迹连续漏检即终止（一闪而过不确认）
    "lost_after_misses": 3,       # 确认航迹连续漏检此次数 → lost
    "terminate_after_misses": 6,  # lost 后累计漏检此次数 → terminated
    "velocity_alpha": 0.5,        # 速度 EMA 系数
    "position_alpha": 0.35,       # 输出角（平滑角）EMA 系数
    "logpower_alpha": 0.3,        # 功率（对数谱高）EMA 系数
    # —— 交叉护航 ——
    "convoy_enter_deg": 8.0,      # 两外推位置近于此值且相向 → 编队
    "convoy_region_deg": 16.0,    # 只在交叉中心 ±此范围内找恢复峰
    "convoy_margin_deg": 1.5,     # 预测越过中心多远才允许认回
    "convoy_wide_deg": 7.0,       # 平移后认回残差上限
    "convoy_release_deg": 11.0,   # 两峰分开到此值以上解除护航
    "convoy_max_age": 40,         # 护航最多次数；超时放弃保护回普通逻辑
    "grace_scans": 6,             # 护航解除后抑制立即重新编队的次数
}


@dataclass
class TrackPoint:
    """航迹历史上的一个点。

    ``kind="measurement"`` 是关联到的实测峰；``kind="coast"`` 是交叉
    护航期间没有可信峰时按运动趋势外推的占位（角度仍给出，供上位机
    连续显示；本服务会明确标注它不是实测）。
    """

    version: int
    angle: float
    kind: PointKind


@dataclass(eq=False)
class Track:
    id: int
    state: TrackState
    angle: float
    smoothed_angle: float
    velocity_deg_per_scan: float
    consecutive_hits: int
    consecutive_misses: int
    log_power: float | None
    created_version: int
    last_seen_version: int
    history: list[TrackPoint] = field(default_factory=list)
    group_id: int | None = field(default=None, repr=False)
    grace: int = field(default=0, repr=False)

    def to_dict(self) -> dict:
        return {
            "track_id": self.id,
            "state": self.state,
            "angle": self.angle,
            "smoothed_angle": self.smoothed_angle,
            "velocity_deg_per_scan": self.velocity_deg_per_scan,
            "consecutive_hits": self.consecutive_hits,
            "consecutive_misses": self.consecutive_misses,
            "created_version": self.created_version,
            "last_seen_version": self.last_seen_version,
        }


@dataclass
class TrackEvent:
    """一次航迹生命周期变化（绑定到产生它的估计版本号）。"""

    version: int
    event: EventType
    track_id: int
    state: TrackState
    angle: float | None = None

    def to_dict(self) -> dict:
        out = {
            "version": self.version,
            "event": self.event,
            "track_id": self.track_id,
            "state": self.state,
        }
        if self.angle is not None:
            out["angle"] = self.angle
        return out


@dataclass
class Detection:
    """喂给跟踪器的一次峰观测。"""

    angle: float
    log_power: float
    detection_index: int  # 在本次 MUSIC 结果 angles 里的下标


@dataclass
class AngleAssignment:
    """某个峰的归属结论。"""

    detection_index: int
    angle: float
    track_id: int | None
    associated: bool
    reason: AssignReason


@dataclass
class UpdateResult:
    assignments: list[AngleAssignment]
    events: list[TrackEvent]


class _Convoy:
    """交叉护航组（恰好两条确认航迹）。"""

    def __init__(self, group_id: int, tracks: list[Track], center: float,
                 dirs: dict[int, int]):
        self.id = group_id
        self.tracks = tracks
        self.center = center
        # Track.id -> 进入护航时冻结的运动方向（+1/-1）
        self.dirs = dirs
        self.phase: Literal["approach", "assigned"] = "approach"
        self.age = 0


def _params(overrides: dict | None) -> dict:
    p = dict(DEFAULT_PARAMS)
    if overrides:
        for k, v in overrides.items():
            if k in p and v is not None:
                p[k] = v
    return p


class AngleTracker:
    """逐次估计的角度航迹跟踪器。

    纯状态机：:meth:`update` 输入本次检测，内部状态推进，输出归属与事件。
    状态可通过 :meth:`to_state` / :meth:`from_state` 完整序列化，落盘后
    重启续写与一口气跑完完全一致（状态不含时间戳/随机数）。
    """

    def __init__(self, params: dict | None = None):
        self._p = _params(params)
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._next_group_id = 1
        self._groups: dict[int, _Convoy] = {}

    # ------------------------------------------------------------------
    def update(self, version: int, detections: list[Detection]) -> UpdateResult:
        """用第 ``version`` 次估计的检测推进航迹。

        每个 detection 恰好被处理一次：要么归属某条航迹（associated），
        要么标记为暂无归属（unassigned / crossing）。
        """
        events: list[TrackEvent] = []
        owners: dict[int, int] = {}   # detection_index -> track_id
        used: set[int] = set()
        active = [t for t in self._tracks.values() if t.state != "terminated"]

        # grace 倒计时（护航解除后短时间内不许重新编队）
        for t in active:
            if t.group_id is None and t.grace > 0:
                t.grace -= 1

        # 1) 编队：两条确认、相向、外推位置足够近的航迹组成护航组
        self._maybe_form_convoy(active)

        # 2) 护航组：approach 外推 / 恢复认回 / assigned 持续认回。
        # 注意：组可能在本帧解除（dissolve），必须在处理*前*固定本帧归属
        # 护航的航迹集合，否则解除后这些航迹同一帧又会进普通 GNN 被关联两次。
        convoy_track_ids = {
            tid
            for gg in self._groups.values()
            for tid in (x.id for x in gg.tracks)
        }
        convoy_alive = False
        for g in list(self._groups.values()):
            mapping = self._update_convoy(g, version, detections, events)
            for t, det in mapping.items():
                owners[det.detection_index] = t.id
                used.add(det.detection_index)
            if g.id in self._groups:
                convoy_alive = True

        # 3) 普通 GNN 关联（本帧开始时不在任何护航组里的航迹）
        normal = [t for t in active if t.id not in convoy_track_ids]
        assigned_det = self._gnn(normal, detections)
        for t, det in assigned_det.items():
            self._apply_hit(t, det.angle, det.log_power, version, events)
            owners[det.detection_index] = t.id
            used.add(det.detection_index)

        # 4) 普通航迹漏检 → 生命周期推进
        for t in normal:
            if t not in assigned_det:
                t.consecutive_misses += 1
                self._apply_miss(t, version, events)

        # 5) 剩余峰：护航未决期间不允许产生新航迹（假峰防护）
        for det in detections:
            if det.detection_index in used:
                continue
            if convoy_alive:
                continue
            tid = self._birth(det, version, events)
            owners[det.detection_index] = tid
            used.add(det.detection_index)

        # 6) 组装归属（保持检测原始顺序）
        assignments: list[AngleAssignment] = []
        for det in detections:
            tid = owners.get(det.detection_index)
            if tid is not None:
                assignments.append(AngleAssignment(
                    det.detection_index, det.angle, tid, True, "associated"
                ))
            elif convoy_alive:
                assignments.append(AngleAssignment(
                    det.detection_index, det.angle, None, False, "crossing"
                ))
            else:
                assignments.append(AngleAssignment(
                    det.detection_index, det.angle, None, False, "unassigned"
                ))
        return UpdateResult(assignments=assignments, events=events)

    # ------------------------------------------------------------------
    # 普通关联
    # ------------------------------------------------------------------
    def _gate(self, t: Track) -> float:
        return (self._p["tentative_gate_deg"] if t.state == "tentative"
                else self._p["confirm_gate_deg"])

    def _predict(self, t: Track) -> float:
        return t.angle + t.velocity_deg_per_scan

    def _gnn(self, tracks: list[Track],
             detections: list[Detection]) -> dict[Track, Detection]:
        """全局最近邻：所有 (航迹,波门内峰) 候选按残差排序，贪心唯一匹配。"""
        cands: list[tuple[float, int, int]] = []
        for t in tracks:
            gate = self._gate(t)
            pred = self._predict(t)
            for det in detections:
                d = abs(det.angle - pred)
                if d <= gate:
                    cands.append((d, t.id, det.detection_index))
        cands.sort(key=lambda x: x[0])
        det_by_idx = {d.detection_index: d for d in detections}
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        out: dict[Track, Detection] = {}
        for _, tid, didx in cands:
            if tid in used_tracks or didx in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(didx)
            track = self._tracks[tid]
            out[track] = det_by_idx[didx]
        return out

    def _apply_hit(self, t: Track, angle: float, log_power: float,
                   version: int, events: list[TrackEvent]) -> None:
        p = self._p
        delta = angle - t.angle
        if t.consecutive_hits < 1:
            t.velocity_deg_per_scan = delta
        else:
            a = float(p["velocity_alpha"])
            t.velocity_deg_per_scan = (1 - a) * t.velocity_deg_per_scan + a * delta
        # raw 角用于关联/护航/速度；smoothed 角是航迹对外的位置与历史点
        if t.consecutive_hits < 1:
            t.smoothed_angle = angle
        else:
            b = float(p["position_alpha"])
            t.smoothed_angle = (1 - b) * t.smoothed_angle + b * angle
        t.angle = angle
        if t.log_power is None:
            t.log_power = log_power
        else:
            c = float(p["logpower_alpha"])
            t.log_power = (1 - c) * t.log_power + c * log_power
        t.consecutive_hits += 1
        t.consecutive_misses = 0
        t.last_seen_version = version
        t.history.append(TrackPoint(version, t.smoothed_angle, "measurement"))
        if t.state == "tentative" and t.consecutive_hits >= int(p["confirm_hits"]):
            t.state = "confirmed"
            events.append(TrackEvent(version, "track_confirmed", t.id, t.state,
                                     t.smoothed_angle))
        elif t.state == "lost":
            t.state = "confirmed"
            events.append(TrackEvent(version, "track_reacquired", t.id, t.state,
                                     t.smoothed_angle))

    def _apply_miss(self, t: Track, version: int,
                    events: list[TrackEvent]) -> None:
        p = self._p
        if t.state == "tentative" and t.consecutive_misses >= int(p["tentative_max_misses"]):
            t.state = "terminated"
            events.append(TrackEvent(version, "track_terminated", t.id, t.state))
        elif t.state == "confirmed" and t.consecutive_misses >= int(p["lost_after_misses"]):
            t.state = "lost"
            events.append(TrackEvent(version, "track_lost", t.id, t.state))
        elif t.state == "lost" and t.consecutive_misses >= int(p["terminate_after_misses"]):
            t.state = "terminated"
            events.append(TrackEvent(version, "track_terminated", t.id, t.state))

    def _birth(self, det: Detection, version: int,
               events: list[TrackEvent]) -> int:
        tid = self._next_id
        self._next_id += 1
        t = Track(
            id=tid,
            state="tentative",
            angle=det.angle,
            smoothed_angle=det.angle,
            velocity_deg_per_scan=0.0,  # 首点速度未知：用当前位置外推
            consecutive_hits=1,
            consecutive_misses=0,
            log_power=det.log_power,
            created_version=version,
            last_seen_version=version,
            history=[TrackPoint(version, det.angle, "measurement")],
        )
        self._tracks[tid] = t
        events.append(TrackEvent(version, "track_created", tid, "tentative", det.angle))
        return tid

    # ------------------------------------------------------------------
    # 交叉护航
    # ------------------------------------------------------------------
    def _maybe_form_convoy(self, active: list[Track]) -> None:
        enter = float(self._p["convoy_enter_deg"])
        free = [
            t for t in active
            if t.state == "confirmed" and t.group_id is None and t.grace == 0
        ]
        for i in range(len(free)):
            for j in range(i + 1, len(free)):
                x, y = free[i], free[j]
                if x.velocity_deg_per_scan * y.velocity_deg_per_scan >= 0:
                    continue  # 必须相向
                if abs(self._predict(x) - self._predict(y)) > enter:
                    continue
                gid = self._next_group_id
                self._next_group_id += 1
                center = 0.5 * (self._predict(x) + self._predict(y))
                dirs = {
                    x.id: int(math.copysign(1, x.velocity_deg_per_scan)),
                    y.id: int(math.copysign(1, y.velocity_deg_per_scan)),
                }
                g = _Convoy(gid, [x, y], center, dirs)
                self._groups[gid] = g
                x.group_id = gid
                y.group_id = gid
                return

    def _update_convoy(self, g: _Convoy, version: int,
                       detections: list[Detection],
                       events: list[TrackEvent]) -> dict[Track, Detection]:
        """推进一个护航组。

        返回 ``{航迹: 被吃掉的检测}``：空字典表示本帧两航迹都在按运动
        趋势外推（coast），不归任何实测峰。
        """
        p = self._p
        g.age += 1
        region = float(p["convoy_region_deg"])
        in_region = sorted(
            (d for d in detections if abs(d.angle - g.center) <= region),
            key=lambda d: d.angle,
        )
        left = [d for d in in_region if d.angle < g.center]
        right = [d for d in in_region if d.angle > g.center]

        mapping: dict[Track, Detection] = {}
        if left and right:
            lo, hi = left[-1], right[0]  # 最靠近中心的两侧各一峰
            candidate = {t: (hi if g.dirs[t.id] > 0 else lo) for t in g.tracks}
            if g.phase == "approach":
                margin = float(p["convoy_margin_deg"])
                crossed = all(
                    g.dirs[t.id] * (self._predict(t) - g.center) > margin
                    for t in g.tracks
                )
                if crossed:
                    shift = (
                        0.5 * (lo.angle + hi.angle)
                        - 0.5 * sum(self._predict(t) for t in g.tracks)
                    )
                    wide = float(p["convoy_wide_deg"])
                    if all(
                        abs(candidate[t].angle - (self._predict(t) + shift)) <= wide
                        for t in g.tracks
                    ):
                        mapping = candidate
                        g.phase = "assigned"
            else:  # assigned：按冻结方向认回本侧最近峰
                mapping = candidate

        if len(mapping) == 2:
            for t, det in mapping.items():
                self._apply_hit(t, det.angle, det.log_power, version, events)
            span = max(d.angle for d in mapping.values()) - min(
                d.angle for d in mapping.values()
            )
            if g.phase == "assigned" and span >= float(p["convoy_release_deg"]):
                self._dissolve(g)
            return mapping

        # 无两峰可认：按各自速度外推，coast 点不计漏检。
        # raw 角与 smoothed 角都按同一速度推进，平滑角作为历史点输出，
        # 保证交叉期间航迹连续、且与恢复后的平滑序列衔接。
        for t in g.tracks:
            step = t.velocity_deg_per_scan
            t.angle = t.angle + step
            t.smoothed_angle = t.smoothed_angle + step
            t.history.append(TrackPoint(version, t.smoothed_angle, "coast"))
        if g.age > int(p["convoy_max_age"]):
            # 护航超时（两目标长时间分不开的极端情形）：放弃保护回普通
            # 逻辑，本帧不计漏检；给 grace 窗口避免立刻重新编队。
            self._dissolve(g, grace=int(p["grace_scans"]))
        return {}

    def _dissolve(self, g: _Convoy, grace: int = 0) -> None:
        for t in g.tracks:
            t.group_id = None
            t.grace = grace
        self._groups.pop(g.id, None)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def list_tracks(self, include_terminated: bool = True) -> list[Track]:
        out = list(self._tracks.values())
        if not include_terminated:
            out = [t for t in out if t.state != "terminated"]
        return sorted(out, key=lambda t: t.id)

    def get_track(self, track_id: int) -> Track | None:
        return self._tracks.get(track_id)

    def history_of(self, track_id: int) -> list[TrackPoint] | None:
        t = self._tracks.get(track_id)
        return list(t.history) if t is not None else None

    # ------------------------------------------------------------------
    # 序列化（落盘；不含时间戳，保证重启逐项一致）
    # ------------------------------------------------------------------
    def to_state(self) -> dict:
        tracks = []
        for t in self._tracks.values():
            tracks.append({
                "id": t.id,
                "state": t.state,
                "angle": t.angle,
                "smoothed_angle": t.smoothed_angle,
                "velocity": t.velocity_deg_per_scan,
                "hits": t.consecutive_hits,
                "misses": t.consecutive_misses,
                "log_power": t.log_power,
                "created_version": t.created_version,
                "last_seen_version": t.last_seen_version,
                "grace": t.grace,
                "history": [
                    {"version": pt.version, "angle": pt.angle, "kind": pt.kind}
                    for pt in t.history
                ],
            })
        groups = [
            {
                "id": g.id,
                "track_ids": [t.id for t in g.tracks],
                "center": g.center,
                "dirs": {str(t.id): g.dirs[t.id] for t in g.tracks},
                "phase": g.phase,
                "age": g.age,
            }
            for g in self._groups.values()
        ]
        return {
            "version": 3,
            "params": self._p,
            "next_id": self._next_id,
            "next_group_id": self._next_group_id,
            "tracks": tracks,
            "groups": groups,
        }

    @classmethod
    def from_state(cls, state: dict) -> "AngleTracker":
        tracker = cls(params=state.get("params"))
        tracker._next_id = int(state["next_id"])
        tracker._next_group_id = int(state.get("next_group_id", 1))
        for raw in state.get("tracks", []):
            t = Track(
                id=int(raw["id"]),
                state=raw["state"],
                angle=float(raw["angle"]),
                smoothed_angle=float(raw.get("smoothed_angle", raw["angle"])),
                velocity_deg_per_scan=float(raw["velocity"]),
                consecutive_hits=int(raw["hits"]),
                consecutive_misses=int(raw["misses"]),
                log_power=raw.get("log_power"),
                created_version=int(raw["created_version"]),
                last_seen_version=int(raw["last_seen_version"]),
                grace=int(raw.get("grace", 0)),
                history=[
                    TrackPoint(int(h["version"]), float(h["angle"]), h["kind"])
                    for h in raw.get("history", [])
                ],
            )
            tracker._tracks[t.id] = t
        for graw in state.get("groups", []):
            tids = [int(x) for x in graw["track_ids"]]
            gtracks = [tracker._tracks[i] for i in tids]
            dirs = {
                int(k): int(v) for k, v in graw["dirs"].items()
            }
            g = _Convoy(int(graw["id"]), gtracks, float(graw["center"]), dirs)
            g.phase = graw.get("phase", "approach")
            g.age = int(graw.get("age", 0))
            tracker._groups[g.id] = g
            for t in gtracks:
                t.group_id = g.id
        return tracker

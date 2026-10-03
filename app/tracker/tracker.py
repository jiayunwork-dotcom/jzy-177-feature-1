"""来波方向航迹关联：预测 + 全局最近邻 + 交叉保持。

输入是每次估计得到的一组（已按角度排序的）测量：角度与逐角波束功率；
输出是每个测量的归属、航迹快照与状态变化事件。模块不接触存储、HTTP、
时间戳，全部更新是确定性函数，因此“重启后重放”与“一口气跑完”在
相同输入序列下逐项一致。

航迹生命周期::

    tentative ──连续命中 CONFIRM_HITS 次──▶ confirmed
    tentative ──连续 TENTATIVE_MAX_MISSES 次未关联──▶ terminated
    confirmed ──连续 LOST_AFTER 次未关联──▶ lost
    lost      ──重新命中──▶ confirmed（reacquired）
    confirmed/lost ──连续 TERMINATE_AFTER 次未关联──▶ terminated

交叉处理（本模块相对纯最近邻关联多出来的唯一机制，详见设计文档）：

* **成组触发**：两条已确认航迹的一步预测位置间距 ≤ ``MERGE_ENTER_DEG``
  直接成组；或间距 ≤ ``NEAR_DEG`` 且在相互靠近、同时组邻域内的测量数
  少于航迹数（MDL/谱峰已经少报）也成组。
* **组内保持（coasting）**：组成员不参与普通关联、不计丢失，组邻域内
  的测量不归属、也不生成新航迹（交叉附近的旁瓣假峰就此被隔离）。
* **按运动趋势解组**：每个成员都在 ``RESOLVE_GATE_DEG`` 内找到各自的
  测量、测量彼此分开 ≥ ``RESOLVE_MIN_SEP_DEG``，且“分给我比分给对方
  至少便宜 ``RESOLVE_MARGIN_DEG``”时才按预测续接解组；功率对数差作为
  同分时的次级排序。条件不满足就继续等，长等有 ``MAX_GROUP_HOLD`` 兜底。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import permutations

# ---- 生命周期门限（每次估计计一拍）-------------------------------------
CONFIRM_HITS = 3            # 连续命中这么多拍才确认（一闪而过不可确认）
TENTATIVE_MAX_MISSES = 2    # 候选航迹连续这么多拍没见到即终止
LOST_AFTER = 3              # 确认航迹连续这么多拍没见到 -> 暂时丢失
TERMINATE_AFTER = 6         # 连续这么多拍没见到 -> 终止

# ---- 关联门限（度）------------------------------------------------------
GATE_DEG = 3.5              # 普通航迹关联门
LOST_GATE_DEG = 8.0         # 丢失航迹重获门（放宽）
MERGE_ENTER_DEG = 2.5       # 预测间距小于此值直接进入交叉保持
NEAR_DEG = 6.0              # “靠近”判定与组邻域半径
RESOLVE_GATE_DEG = 3.5      # 解组时测量距预测的最大允许偏差
RESOLVE_MIN_SEP_DEG = 2.5   # 解组要求测量之间至少分开这么多
RESOLVE_MARGIN_DEG = 1.0    # 归属相对另一成员必须便宜这么多（防互换）
MAX_GROUP_HOLD = 12         # 交叉保持最多持续多少拍，之后强制释放
VELOCITY_WINDOW = 5         # 速度由最近这么多个点最小二乘给出
POWER_EMA = 0.5             # 功率指数平滑系数
SPAWN_POWER_RATIO = 0.2     # 新航迹功率须 ≥ 现存航迹最大功率的该比例

TRACK_STATUS = ("tentative", "confirmed", "lost", "terminated")


@dataclass
class Track:
    track_id: int
    status: str = "tentative"
    hits: int = 0
    misses: int = 0
    # 每个已归属测量一个点：{"version", "angle", "power"}
    history: list[dict] = field(default_factory=list)
    power_smooth: float | None = None
    first_version: int | None = None
    last_hit_version: int | None = None

    @property
    def last_angle(self) -> float | None:
        return self.history[-1]["angle"] if self.history else None

    def velocity(self) -> float:
        """最近 ``VELOCITY_WINDOW`` 个点的最小二乘斜率（度/每次估计）。"""
        if len(self.history) < 2:
            return 0.0
        pts = self.history[-VELOCITY_WINDOW:]
        v0 = pts[0]["version"]
        xs = [p["version"] - v0 for p in pts]
        ys = [p["angle"] for p in pts]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den <= 0:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den

    def predict(self, version: int, coast_from: float | None = None) -> float:
        """把 ``coast_from``（或最后一次命中角）按速度外推到 ``version``。"""
        base = coast_from if coast_from is not None else self.last_angle
        if base is None or self.last_hit_version is None:
            return 0.0
        return base + self.velocity() * (version - self.last_hit_version)

    def snapshot(self) -> dict:
        return {
            "track_id": self.track_id,
            "status": self.status,
            "angle": self.last_angle,
            "power": self.power_smooth,
            "velocity_deg_per_update": round(self.velocity(), 6),
            "hits": self.hits,
            "consecutive_misses": self.misses,
            "history_points": len(self.history),
            "first_version": self.first_version,
            "last_hit_version": self.last_hit_version,
        }

    def to_state(self) -> dict:
        return {
            "track_id": self.track_id,
            "status": self.status,
            "hits": self.hits,
            "misses": self.misses,
            "history": self.history,
            "power_smooth": self.power_smooth,
            "first_version": self.first_version,
            "last_hit_version": self.last_hit_version,
        }

    @classmethod
    def from_state(cls, d: dict) -> "Track":
        t = cls(
            track_id=int(d["track_id"]),
            status=d["status"],
            hits=int(d["hits"]),
            misses=int(d["misses"]),
            history=list(d["history"]),
            power_smooth=d.get("power_smooth"),
            first_version=d.get("first_version"),
            last_hit_version=d.get("last_hit_version"),
        )
        return t


@dataclass
class _Group:
    group_id: int
    member_ids: list[int]
    formed_version: int
    hold_count: int = 0
    # 每个成员在保持期间持续外推的角度（不写进航迹历史）
    coast_angles: dict[int, float] = field(default_factory=dict)

    def to_state(self) -> dict:
        return {
            "group_id": self.group_id,
            "member_ids": list(self.member_ids),
            "formed_version": self.formed_version,
            "hold_count": self.hold_count,
            "coast_angles": {str(k): v for k, v in self.coast_angles.items()},
        }

    @classmethod
    def from_state(cls, d: dict) -> "_Group":
        return cls(
            group_id=int(d["group_id"]),
            member_ids=[int(x) for x in d["member_ids"]],
            formed_version=int(d["formed_version"]),
            hold_count=int(d.get("hold_count", 0)),
            coast_angles={int(k): float(v) for k, v in d.get("coast_angles", {}).items()},
        )


class Tracker:
    """会话级航迹关联器（无锁；并发由会话层的锁串行化保证）。"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._tracks: dict[int, Track] = {}
        self._groups: list[_Group] = []
        self._next_id = 1
        self._next_group_id = 1

    # ------------------------------------------------------------------
    def list_tracks(self, include_terminated: bool = True) -> list[Track]:
        out = [t for t in self._tracks.values()
               if include_terminated or t.status != "terminated"]
        return sorted(out, key=lambda t: t.track_id)

    def get_track(self, track_id: int) -> Track:
        return self._tracks[track_id]

    # ------------------------------------------------------------------
    def update(self, version: int, angles: list[float],
               powers: list[float]) -> dict:
        """用一次估计的测量（角度、逐角功率）推进航迹。"""
        events: list[dict] = []
        measurements = [
            {"index": i, "angle": float(a), "power": float(p)}
            for i, (a, p) in enumerate(zip(angles, powers))
        ]
        used: set[int] = set()
        hit_this_update: set[int] = set()

        active = [t for t in self._tracks.values() if t.status != "terminated"]
        preds = {t.track_id: t.predict(version) for t in active}

        # 1) 交叉成组（含上一拍遗留组的维护） ----------------------------
        self._detect_groups(version, active, preds, measurements, events)
        coasting: set[int] = set()
        for g in self._groups:
            coasting.update(g.member_ids)

        # 2) 尝试按运动趋势解组 ------------------------------------------
        for g in list(self._groups):
            resolved = self._try_resolve(version, g, measurements, used,
                                         hit_this_update, preds, events)
            if resolved:
                coasting.difference_update(g.member_ids)

        # 3) 普通全局最近邻关联（贪心，按代价全局排序） -------------------
        assignments: dict[int, int] = {}
        candidates = []
        for tid, pred in preds.items():
            if tid in coasting:
                continue
            track = self._tracks[tid]
            gate = LOST_GATE_DEG if track.status == "lost" else GATE_DEG
            for j, m in enumerate(measurements):
                if j in used:
                    continue
                dist = abs(m["angle"] - pred)
                if dist <= gate:
                    p_ref = track.power_smooth
                    p_term = (
                        0.2 * abs(math.log(max(m["power"], 1e-300)
                                           / max(p_ref or m["power"], 1e-300)))
                    )
                    candidates.append((dist + p_term, dist, tid, j))
        candidates.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
        assigned_tracks: set[int] = set()
        for _, _, tid, j in candidates:
            if tid in assigned_tracks or j in used:
                continue
            assigned_tracks.add(tid)
            used.add(j)
            assignments[j] = tid
            hit_this_update.add(tid)
            self._register_hit(version, self._tracks[tid], measurements[j],
                               events)

        # 4) 未关联测量 -> 新候选航迹（交叉邻域内 / 太弱的不生成） -------
        regions = self._group_regions(preds)
        max_power = max(
            (t.power_smooth for t in active if t.power_smooth is not None),
            default=None,
        )
        for j, m in enumerate(measurements):
            if j in used:
                continue
            if any(abs(m["angle"] - c) <= NEAR_DEG for c in regions):
                continue
            if max_power is not None and m["power"] < SPAWN_POWER_RATIO * max_power:
                continue
            new_track = self._spawn(version, m, events)
            # 这个测量正是新航迹的来源：当拍即归属（只发一次，不回改）
            hit_this_update.add(new_track.track_id)
            used.add(j)
            assignments[j] = new_track.track_id

        # 5) 未关联航迹：计丢失 / 保持推进 / 强制释放 --------------------
        for tid, track in self._tracks.items():
            if track.status == "terminated" or tid in hit_this_update:
                continue
            if tid in coasting:
                self._advance_coast(version, tid)
                continue
            self._register_miss(version, track, events)

        for g in list(self._groups):
            g.hold_count += 1
            if g.hold_count >= MAX_GROUP_HOLD:
                # 兜底：长时间解不开就强制释放，交回普通门限关联
                events.append({
                    "type": "crossing_force_released",
                    "version": version,
                    "group_id": g.group_id,
                    "track_ids": list(g.member_ids),
                })
                coasting.difference_update(g.member_ids)
                self._groups.remove(g)

        return {
            "assignments": self._build_assignments(measurements, assignments,
                                                   coasting),
            "tracks": [t.snapshot() for t in self.list_tracks()],
            "events": events,
            "crossing": {
                "active": len(self._groups) > 0,
                "group_ids": [g.group_id for g in self._groups],
                "coasting_track_ids": sorted(coasting),
            },
        }

    # ------------------------------------------------------------------
    def _detect_groups(self, version: int, active: list[Track],
                       preds: dict[int, float],
                       measurements: list[dict],
                       events: list[dict]) -> None:
        """合并遗留组；并按“预测重合 / 靠近且少报”发现新组。"""
        grouped: set[int] = set()
        for g in self._groups:
            g.member_ids = [i for i in g.member_ids
                            if i in self._tracks
                            and self._tracks[i].status != "terminated"]
            grouped.update(g.member_ids)
        self._groups = [g for g in self._groups if g.member_ids]

        eligible = [
            t for t in active
            if t.track_id not in grouped and t.status == "confirmed"
        ]
        # 简单的并查集聚类
        parent = {t.track_id: t.track_id for t in eligible}

        def find(x: int) -> int:
            parent[x] = x if parent[x] == x else find(parent[x])
            return parent[x]

        def union(a: int, b: int) -> None:
            parent[find(a)] = find(b)

        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                a, b = eligible[i], eligible[j]
                sep = abs(preds[a.track_id] - preds[b.track_id])
                if sep > NEAR_DEG:
                    continue
                if sep <= MERGE_ENTER_DEG:
                    union(a.track_id, b.track_id)
                    continue
                # 靠近且在相向运动
                pa, pb = preds[a.track_id], preds[b.track_id]
                closing = (pb - pa) * (a.velocity() - b.velocity()) > 0
                if not closing:
                    continue
                mid = 0.5 * (pa + pb)
                n_near = sum(1 for m in measurements
                             if abs(m["angle"] - mid) <= NEAR_DEG)
                if n_near < 2:  # 组邻域内测量数不足 -> 已经少报
                    union(a.track_id, b.track_id)

        clusters: dict[int, list[int]] = {}
        for tid in sorted(parent):
            clusters.setdefault(find(tid), []).append(tid)
        for member_ids in clusters.values():
            if len(member_ids) < 2:
                continue
            gid = self._next_group_id
            self._next_group_id += 1
            group = _Group(
                group_id=gid,
                member_ids=member_ids,
                formed_version=version,
                coast_angles={tid: preds[tid] for tid in member_ids},
            )
            self._groups.append(group)
            events.append({
                "type": "crossing_started",
                "version": version,
                "group_id": gid,
                "track_ids": list(member_ids),
                "predicted_angles": {
                    str(tid): preds[tid] for tid in member_ids
                },
            })

    def _group_regions(self, preds: dict[int, float]) -> list[float]:
        """每个保持中组的邻域中心（成员外推角的均值，各组独立）。"""
        centers = []
        for g in self._groups:
            angles = [
                g.coast_angles.get(tid, preds.get(tid, 0.0))
                for tid in g.member_ids
            ]
            if angles:
                centers.append(sum(angles) / len(angles))
        return centers

    def _advance_coast(self, version: int, tid: int) -> None:
        """交叉保持期间让预测按速度继续外推（不产生历史点、不计丢失）。"""
        track = self._tracks[tid]
        for g in self._groups:
            if tid in g.member_ids:
                base = g.coast_angles.get(tid, track.last_angle)
                if base is not None:
                    g.coast_angles[tid] = base + track.velocity()

    def _try_resolve(self, version: int, group: _Group,
                     measurements: list[dict], used: set[int],
                     hit_this_update: set[int],
                     preds: dict[int, float], events: list[dict]) -> bool:
        """尝试把组成员与测量一一续接；硬条件不满足就继续保持。"""
        tracks = [self._tracks[tid] for tid in group.member_ids]
        n = len(tracks)
        cand_idx = [
            j for j, m in enumerate(measurements)
            if j not in used
            and min(abs(m["angle"] - group.coast_angles.get(
                t.track_id, preds[t.track_id])) for t in tracks)
            <= RESOLVE_GATE_DEG
        ]
        if len(cand_idx) < n:
            return False

        best = None  # (score, assignment[(tid, j)])
        for chosen in permutations(cand_idx, n):
            costs = {}
            ok = True
            for t, j in zip(tracks, chosen):
                pred = group.coast_angles.get(t.track_id, preds[t.track_id])
                d = abs(measurements[j]["angle"] - pred)
                if d > RESOLVE_GATE_DEG:
                    ok = False
                    break
                p_ref = t.power_smooth or measurements[j]["power"]
                p_term = 0.2 * abs(math.log(
                    max(measurements[j]["power"], 1e-300)
                    / max(p_ref, 1e-300)))
                costs[(t.track_id, j)] = d + p_term
            if not ok:
                continue
            angs = [measurements[j]["angle"] for j in chosen]
            if min(abs(angs[a] - angs[b])
                   for a in range(n) for b in range(a + 1, n)) \
                    < RESOLVE_MIN_SEP_DEG:
                continue
            # 防互换：分给我必须比分给组内任一其他成员便宜一个裕量
            for ia, ta in enumerate(tracks):
                for ib, tb in enumerate(tracks):
                    if ia == ib:
                        continue
                    pred_a = group.coast_angles.get(
                        ta.track_id, preds[ta.track_id])
                    own = abs(measurements[chosen[ia]]["angle"] - pred_a)
                    other = abs(measurements[chosen[ib]]["angle"] - pred_a)
                    if own + RESOLVE_MARGIN_DEG > other:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            score = sum(costs[(t.track_id, j)] for t, j in zip(tracks, chosen))
            if best is None or score < best[0]:
                best = (score, list(zip([t.track_id for t in tracks], chosen)))
        if best is None:
            return False

        for tid, j in best[1]:
            used.add(j)
            hit_this_update.add(tid)
            self._register_hit(version, self._tracks[tid], measurements[j],
                               events)
        events.append({
            "type": "crossing_resolved",
            "version": version,
            "group_id": group.group_id,
            "track_ids": list(group.member_ids),
            "assignment": [
                {"track_id": tid,
                 "angle": measurements[j]["angle"],
                 "power": measurements[j]["power"]}
                for tid, j in best[1]
            ],
        })
        self._groups.remove(group)
        return True

    # ------------------------------------------------------------------
    def _spawn(self, version: int, m: dict, events: list[dict]) -> Track:
        track = Track(track_id=self._next_id)
        self._next_id += 1
        track.first_version = version
        self._tracks[track.track_id] = track
        self._register_hit(version, track, m, events)  # 产生 tentative 事件
        return track

    def _register_hit(self, version: int, track: Track, m: dict,
                      events: list[dict]) -> None:
        previous = track.status
        if track.hits == 0:
            events.append({
                "type": "track_appeared",
                "version": version,
                "track_id": track.track_id,
                "angle": m["angle"],
                "power": m["power"],
            })
        elif previous == "lost":
            events.append({
                "type": "track_reacquired",
                "version": version,
                "track_id": track.track_id,
                "angle": m["angle"],
            })
        track.hits += 1
        track.misses = 0
        track.last_hit_version = version
        track.history.append({
            "version": version,
            "angle": m["angle"],
            "power": m["power"],
        })
        if track.power_smooth is None:
            track.power_smooth = m["power"]
        else:
            track.power_smooth = (
                (1 - POWER_EMA) * track.power_smooth + POWER_EMA * m["power"]
            )
        if previous == "tentative" and track.hits >= CONFIRM_HITS:
            track.status = "confirmed"
            events.append({
                "type": "track_confirmed",
                "version": version,
                "track_id": track.track_id,
                "angle": track.last_angle,
            })
        elif previous == "lost":
            track.status = "confirmed"

    def _register_miss(self, version: int, track: Track,
                       events: list[dict]) -> None:
        track.misses += 1
        if track.status == "tentative":
            if track.misses >= TENTATIVE_MAX_MISSES:
                track.status = "terminated"
                events.append({
                    "type": "track_terminated",
                    "version": version,
                    "track_id": track.track_id,
                    "reason": "tentative_not_confirmed",
                })
            return
        if track.misses == LOST_AFTER and track.status == "confirmed":
            track.status = "lost"
            events.append({
                "type": "track_lost",
                "version": version,
                "track_id": track.track_id,
            })
        if track.misses >= TERMINATE_AFTER:
            track.status = "terminated"
            events.append({
                "type": "track_terminated",
                "version": version,
                "track_id": track.track_id,
                "reason": "missed_too_long",
            })

    def _build_assignments(self, measurements, assignments, coasting):
        out = []
        for j, m in enumerate(measurements):
            tid = assignments.get(j)
            out.append({
                "angle_index": j,
                "angle": m["angle"],
                "power": m["power"],
                "track_id": tid,
                "associated": tid is not None,
            })
        return out

    # ---- 持久化 --------------------------------------------------------
    def to_state(self) -> dict:
        return {
            "enabled": self.enabled,
            "next_id": self._next_id,
            "next_group_id": self._next_group_id,
            "tracks": [t.to_state() for t in self.list_tracks()],
            "groups": [g.to_state() for g in self._groups],
        }

    def load_state(self, state: dict) -> None:
        self.enabled = bool(state.get("enabled", True))
        self._next_id = int(state.get("next_id", 1))
        self._next_group_id = int(state.get("next_group_id", 1))
        self._tracks = {
            int(d["track_id"]): Track.from_state(d)
            for d in state.get("tracks", [])
        }
        self._groups = [
            _Group.from_state(d) for d in state.get("groups", [])
        ]

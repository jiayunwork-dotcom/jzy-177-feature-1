"""跟踪器纯单元测试：关联、生命周期、确定性、序列化（不经过 HTTP）。"""

from __future__ import annotations

import copy

import pytest

from app.sessions.tracker import (
    AngleTracker,
    DEFAULT_PARAMS,
    Detection,
)


def _dets(angles, log_power=0.0):
    return [Detection(float(a), float(log_power), i)
            for i, a in enumerate(angles)]


def _states(tr):
    return {t.id: t.state for t in tr.list_tracks()}


def test_three_hits_confirm_then_miss_lifecycle():
    tr = AngleTracker()
    # 连续 3 次同一位置 → confirmed
    for v in range(1, 4):
        r = tr.update(v, _dets([10.0]))
    assert set(_states(tr).values()) == {"confirmed"}
    track = tr.get_track(1)
    assert track.consecutive_hits == 3
    # 连续 3 次没峰 → lost；再 3 次（累计 6）→ terminated
    for v in range(4, 7):
        tr.update(v, _dets([]))
    assert tr.get_track(1).state == "lost"
    for v in range(7, 10):
        tr.update(v, _dets([]))
    assert tr.get_track(1).state == "terminated"


def test_tentative_dies_after_two_misses_no_confirmation():
    tr = AngleTracker()
    tr.update(1, _dets([30.0]))           # 起 tentative
    assert _states(tr)[1] == "tentative"
    tr.update(2, _dets([]))
    tr.update(3, _dets([]))               # 连续 2 次漏检
    assert _states(tr)[1] == "terminated"


def test_lost_track_reacquired_within_gate():
    tr = AngleTracker()
    for v in range(1, 4):
        tr.update(v, _dets([0.0]))
    tr.update(4, _dets([]))
    tr.update(5, _dets([]))
    tr.update(6, _dets([]))
    assert tr.get_track(1).state == "lost"  # 连续 3 次漏检
    res = tr.update(7, _dets([1.0]))        # 在确认波门内重现
    assert tr.get_track(1).state == "confirmed"
    assert any(e.event == "track_reacquired" for e in res.events)


def test_gnn_unique_one_detection_per_track_per_scan():
    tr = AngleTracker()
    # 建立两条分开的确认航迹
    for v in range(1, 4):
        tr.update(v, _dets([-20.0, 20.0]))
    # 两条航迹挤到同一个峰附近：只有一条能吃到，另一条计漏检，绝不共享
    res = tr.update(4, _dets([0.0]))
    assoc = [a for a in res.assignments if a.associated]
    assert len(assoc) == 1
    owners = [a.track_id for a in assoc]
    assert len(owners) == len(set(owners)) == 1


def test_unassigned_detection_starts_tentative():
    tr = AngleTracker()
    res = tr.update(1, _dets([-40.0, 40.0]))
    assert {a.track_id for a in res.assignments if a.associated} == {1, 2}
    assert all(a.reason == "associated" for a in res.assignments)


def test_update_is_deterministic_and_serialization_roundtrips():
    """同样检测序列 → 同样结果；中途序列化往返后继续跑 == 一口气跑完。"""
    def feed(tr, detections_seq, start=1):
        for k, det in enumerate(detections_seq):
            tr.update(start + k, det)

    seq = [
        _dets([-15.0 + 30.0 * i / 39.0, 15.0 - 30.0 * i / 39.0])
        for i in range(40)
    ]
    ref = AngleTracker()
    feed(ref, copy.deepcopy(seq))

    # 在交叉中段（护航进行中）序列化、恢复，再继续推进
    cut = AngleTracker()
    feed(cut, copy.deepcopy(seq[:17]))
    cut = AngleTracker.from_state(cut.to_state())
    feed(cut, copy.deepcopy(seq[17:]), start=18)
    assert cut.to_state() == ref.to_state()

    # 完全独立的两次跑也逐项一致（纯确定性）
    a = AngleTracker()
    b = AngleTracker()
    feed(a, copy.deepcopy(seq))
    feed(b, copy.deepcopy(seq))
    assert a.to_state() == b.to_state()


def test_default_params_present():
    for key in ("tentative_gate_deg", "confirm_gate_deg", "convoy_enter_deg",
                "convoy_release_deg", "position_alpha"):
        assert key in DEFAULT_PARAMS


def test_params_overrides_apply():
    tr = AngleTracker({"confirm_hits": 5, "tentative_gate_deg": 2.0})
    assert tr._p["confirm_hits"] == 5
    assert tr._p["tentative_gate_deg"] == 2.0
    # 未覆盖项保留默认
    assert tr._p["confirm_gate_deg"] == DEFAULT_PARAMS["confirm_gate_deg"]


def test_smoothed_angle_distinct_from_raw_and_recorded_in_history():
    tr = AngleTracker({"position_alpha": 0.5})
    tr.update(1, _dets([0.0]))
    tr.update(2, _dets([1.0]))
    t = tr.get_track(1)
    assert t.angle == pytest.approx(1.0)               # 原始角=最新实测
    assert t.smoothed_angle == pytest.approx(0.5)      # 平滑角=0/1 均值
    assert t.history[-1].angle == pytest.approx(0.5)   # 历史记录平滑角

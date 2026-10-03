"""航迹关联内核的直接单测（不过 HTTP/会话）。"""

from __future__ import annotations

import math

from app.tracker import Tracker


def test_empty_measurement_update_is_noop():
    t = Tracker()
    out = t.update(1, [], [])
    assert out["assignments"] == []
    assert out["events"] == []
    assert out["crossing"]["active"] is False
    assert t.list_tracks() == []


def test_spawned_track_is_not_counted_as_miss_on_birth_update():
    """出生那拍本身是命中，不能同拍又计一次未命中（否则误提前终止）。"""
    t = Tracker()
    t.update(1, [10.0], [1.0])
    tr = t.get_track(1)
    assert tr.hits == 1 and tr.misses == 0
    # 随后两拍不见 -> 候选终止（出生拍不算 miss）
    t.update(2, [], [])
    assert t.get_track(1).status == "tentative"
    t.update(3, [], [])
    assert t.get_track(1).status == "terminated"


def test_confirm_lost_reacquire_terminate_sequence():
    t = Tracker()
    for v in range(1, 4):
        t.update(v, [10.0 + 0.01 * v], [1.0])
    assert t.get_track(1).status == "confirmed"
    t.update(4, [], []); t.update(5, [], [])
    assert t.get_track(1).status == "confirmed"  # 2 拍
    t.update(6, [], [])
    assert t.get_track(1).status == "lost"       # 3 拍
    out = t.update(7, [10.3], [1.0])
    assert t.get_track(1).status == "confirmed"
    assert any(e["type"] == "track_reacquired" for e in out["events"])
    assert t.get_track(1).misses == 0
    for v in range(8, 14):
        t.update(v, [], [])
    assert t.get_track(1).status == "terminated"


def test_ids_are_stable_never_reused_and_sorted_by_angle_at_birth():
    t = Tracker()
    out = t.update(1, [30.0, -5.0], [1.0, 1.0])
    # 同拍出生按角度升序编号
    assert [e["track_id"] for e in out["events"]
            if e["type"] == "track_appeared"] == [1, 2]
    for v in range(2, 4):
        t.update(v, [30.0, -5.0], [1.0, 1.0])
    # T1 终止后，新目标拿新编号，1 不复用
    for v in range(4, 10):
        t.update(v, [30.0], [1.0])
    assert t.get_track(2).status == "terminated"
    out = t.update(10, [30.0, 70.0], [1.0, 1.0])
    new_ids = [e["track_id"] for e in out["events"]
               if e["type"] == "track_appeared"]
    assert new_ids == [3]


def test_history_only_records_real_assignments_and_never_rewritten():
    t = Tracker()
    for v, a in [(1, 10.0), (2, 10.1), (3, 10.05)]:
        out = t.update(v, [a], [1.0])
        # 来源测量当拍即归属；归属一经产生不回改
        assert out["assignments"][0]["track_id"] == 1
    h = t.get_track(1).history
    assert [(p["version"], round(p["angle"], 3)) for p in h] == [
        (1, 10.0), (2, 10.1), (3, 10.05)]
    t.update(4, [], [])
    assert len(t.get_track(1).history) == 3  # 缺失拍不补预测点
    # 早期点角度未被后续更新改写
    assert t.get_track(1).history[0]["angle"] == 10.0


def test_weak_sidelobe_peak_does_not_spawn_track():
    """现存航迹 1.0 功率时，0.1 的孤立弱峰不生候选。"""
    t = Tracker()
    for v in range(1, 4):
        t.update(v, [0.0], [1.0])
    out = t.update(4, [0.0, 28.0], [1.0, 0.05])
    assert all(a["track_id"] != 2 for a in out["assignments"])
    assert 2 not in t._tracks


def test_state_roundtrip_is_deterministic():
    t = Tracker()
    seq = [-15, -14, -13, -12, -11, -10, -9, -8, -7, -6,
           -5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    for v, a in enumerate(seq, start=1):
        t.update(v, [float(a)], [1.0])
    state = t.to_state()
    t2 = Tracker()
    t2.load_state(state)
    assert t2.to_state() == state
    out1 = t.update(21, [5.0], [1.0])
    out2 = t2.update(21, [5.0], [1.0])
    assert out1 == out2


def test_three_way_group_permutations_do_not_crash():
    """三条航迹交叉成组：排列枚举正常，门限不满足时保持。"""
    t = Tracker()
    # 三条已确认航迹，从散开位置走到一点
    paths = {
        1: [-30, -27, -24, -21, -18, -15, -12, -9, -6, -3, -1, 0],
        2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        3: [30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 1, 0],
    }
    for v in range(1, 13):
        angs = [paths[i][v - 1] for i in (1, 2, 3)]
        t.update(v, [float(a) for a in angs], [1.0, 1.0, 1.0])
    # 没崩、没有航迹被判丢失终止
    for tid in (1, 2, 3):
        assert t.get_track(tid).status in ("confirmed", "tentative")

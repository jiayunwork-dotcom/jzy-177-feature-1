"""航迹跟踪验收剧本（固定种子，合成接口造数据）。

统一条件：八阵元、半波长间距；每批 50 条快拍、滑动窗口取 50；
每推一批出一次估计。覆盖：静止双目标、中途进出、一闪而过、交叉、
重启逐项一致、并发等价串行回放 + 重复批次无事件。
"""

from __future__ import annotations

import copy
import threading

import numpy as np
import pytest

from tests.conftest import synthetic_batch, tracking_session_config

M = 8
K = 50
# 各剧本固定种子（服务实际不接收种子，种子只用于测试侧造数；同剧本同种子）
STATIC_SEED = 1000
ENTER_SEED = 3000
FLASH_SEED = 4000
CROSS_SEED = 2000


def _new_session(client, **kw):
    cfg = tracking_session_config(**kw)
    return client.post("/sessions", json=cfg).json()["session_id"]


def _push(client, sid, idx, angles, powers, snr_db, seed_base, **kw):
    real, imag = synthetic_batch(
        M, angles, powers, snr_db, seed_base + idx, k=K
    )
    resp = client.post(f"/sessions/{sid}/append", json={
        "batch_id": f"b{idx}",
        "real": real.tolist(),
        "imag": imag.tolist(),
        "estimate": True,
    }, **kw)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _tracks(client, sid, include_terminated=True):
    r = client.get(f"/sessions/{sid}/tracks",
                   params={"include_terminated": include_terminated})
    assert r.status_code == 200
    return r.json()["tracks"]


def _history(client, sid, tid):
    r = client.get(f"/sessions/{sid}/tracks/{tid}")
    assert r.status_code == 200
    return r.json()["track"]


def _events(client, sid, since_version=0):
    r = client.get(f"/sessions/{sid}/track-events",
                   params={"since_version": since_version})
    assert r.status_code == 200
    return r.json()["track_events"]


def _event_seq(client, sid):
    """拍平为 [(version, event, track_id)] 的有序事件序列。"""
    out = []
    for row in _events(client, sid):
        for e in row["events"]:
            out.append((row["version"], e["event"], e["track_id"]))
    return out


# ----------------------------------------------------------------------
def test_static_two_targets_two_stable_tracks(client):
    """静止双目标 −20°/+25°、20 dB、30 批：恰好两条确认航迹，编号不变，
    每个历史点与真值差 < 0.1°。"""
    sid = _new_session(client)
    for i in range(30):
        body = _push(client, sid, i, [-20.0, 25.0], [1.0, 1.0], 20.0, STATIC_SEED)
        at = body["result"]["tracking"]["angle_tracks"]
        assert {x["track_id"] for x in at} == {1, 2}
        assert all(x["associated"] for x in at)

    tracks = _tracks(client, sid, include_terminated=False)
    assert len(tracks) == 2
    assert sorted(t["track_id"] for t in tracks) == [1, 2]
    assert all(t["state"] == "confirmed" for t in tracks)

    for tid, truth in ((1, -20.0), (2, 25.0)):
        h = _history(client, sid, tid)
        assert len(h["history"]) == 30
        assert all(p["kind"] == "measurement" for p in h["history"])
        assert all(abs(p["angle"] - truth) < 0.1 for p in h["history"])

    # 事件：两次 created、两次 confirmed，没有 lost/terminated
    seq = _event_seq(client, sid)
    events_by_type = {}
    for _, ev, _ in seq:
        events_by_type[ev] = events_by_type.get(ev, 0) + 1
    assert events_by_type == {"track_created": 2, "track_confirmed": 2}


# ----------------------------------------------------------------------
def test_target_enters_and_exits_lifecycle(client):
    """前 10 批两个源，中间 10 批 +50° 多一个源，最后 10 批消失：
    第三条航迹经历 created→confirmed→lost→terminated，前两条不受影响。"""
    sid = _new_session(client)
    for i in range(30):
        angles = [-20.0, 25.0] + ([50.0] if 10 <= i <= 19 else [])
        powers = [1.0] * len(angles)
        _push(client, sid, i, angles, powers, 20.0, ENTER_SEED)

    active = _tracks(client, sid, include_terminated=False)
    assert sorted(t["track_id"] for t in active) == [1, 2]
    assert all(t["state"] == "confirmed" for t in active)

    third = _history(client, sid, 3)
    assert third["state"] == "terminated"

    t3_events = [(ev, v) for v, ev, tid in _event_seq(client, sid) if tid == 3]
    names = [ev for ev, _ in t3_events]
    assert names == [
        "track_created", "track_confirmed", "track_lost", "track_terminated"
    ]
    # created/confirmed 发生在它在场期间（版本 11..20）
    assert 11 <= t3_events[0][1] <= 20
    assert 11 <= t3_events[1][1] <= 20
    # lost / terminated 在它消失之后
    assert t3_events[2][1] >= 21
    assert t3_events[3][1] >= 21

    # 前两条没有任何丢失/终止事件，且 30 个实测点
    for tid in (1, 2):
        evs = [ev for _, ev, t in _event_seq(client, sid) if t == tid]
        assert evs == ["track_created", "track_confirmed"]
        h = _history(client, sid, tid)
        assert len(h["history"]) == 30
        assert all(p["kind"] == "measurement" for p in h["history"])


# ----------------------------------------------------------------------
def test_single_batch_flash_leaves_no_confirmed_track(client):
    """只有某一批混进一个额外源：不能留下确认航迹。"""
    sid = _new_session(client)
    for i in range(8):
        angles = [-20.0, 25.0] + ([60.0] if i == 5 else [])
        _push(client, sid, i, angles, [1.0] * len(angles), 20.0, FLASH_SEED)

    confirmed = _tracks(client, sid, include_terminated=False)
    assert sorted(t["track_id"] for t in confirmed) == [1, 2]
    # 闪烁源可能短暂生成过 tentative，但绝不能被 confirmed
    seq = _event_seq(client, sid)
    flash = [(v, ev) for v, ev, tid in seq if tid == 3]
    assert all(ev != "track_confirmed" for _, ev in flash)
    # 且最终被终止（编号封存，不再出现）
    assert any(ev == "track_terminated" for _, ev in flash)


# ----------------------------------------------------------------------
def test_crossing_targets_keep_identities(client):
    """两源从 −15°/+15° 匀速相向，40 批后交换到对侧，功率 1 / 0.5，
    15 dB：穿越前后编号跟着运动方向走，不互换；最终仍是那两条。"""
    sid = _new_session(client)
    for i in range(40):
        th1 = -15.0 + 30.0 * i / 39.0
        th2 = 15.0 - 30.0 * i / 39.0
        _push(client, sid, i, [th1, th2], [1.0, 0.5], 15.0, CROSS_SEED)

    tracks = _tracks(client, sid, include_terminated=False)
    by_id = {t["track_id"]: t for t in tracks}
    assert sorted(by_id) == [1, 2]
    t1 = _history(client, sid, 1)
    t2 = _history(client, sid, 2)
    # 编号跟随运动方向：航迹 1 起始 −15°、结束 +15°；航迹 2 相反。
    # 起点用历史平滑角；终点用航迹当前（原始）角——平滑角在匀速段末端
    # 有固定 EMA 滞后，用它判身份会把“滞后”误算成“串号”。
    assert t1["history"][0]["angle"] < -14.0
    assert by_id[1]["angle"] > 14.0
    assert t2["history"][0]["angle"] > 14.0
    assert by_id[2]["angle"] < -14.0
    assert t1["state"] == t2["state"] == "confirmed"

    # 每条航迹 40 个历史点（交叉期间用 coast 占位也计入，且明确标注）
    for t in (t1, t2):
        assert len(t["history"]) == 40
        kinds = {p["kind"] for p in t["history"]}
        assert kinds <= {"measurement", "coast"}
    # 交叉窗口内至少有一次 coast（验证护航确实触发），且 coast 只在中段
    coast_versions = sorted(
        p["version"] for p in t1["history"] if p["kind"] == "coast"
    )
    assert coast_versions and min(coast_versions) >= 12 and max(coast_versions) <= 32

    # 交叉窗口内的估计角若只有一个峰/出现假峰，归属标注为 crossing（无归属），
    # 已推给上位机的归属始终只认 1/2，绝不出现 3+ 编号
    seq_tids = {tid for _, _, tid in _event_seq(client, sid)}
    assert seq_tids == {1, 2}


# ----------------------------------------------------------------------
def _feed_crossing(session_obj, n, seed_base=CROSS_SEED):
    """用 Session 对象（非 HTTP）直接喂交叉剧本前 n 批。"""
    for i in range(n):
        th1 = -15.0 + 30.0 * i / 39.0
        th2 = 15.0 - 30.0 * i / 39.0
        real, imag = synthetic_batch(
            M, [th1, th2], [1.0, 0.5], 15.0, seed_base + i, k=K
        )
        session_obj.append(f"b{i}", real, imag, run_estimate=True)


def _track_snapshot(manager, sid):
    """与重启无关的可比较航迹快照（不含时间戳）。"""
    s = manager.get(sid)
    tracks = {}
    for t in s.list_tracks(include_terminated=True):
        h = s.get_track_history(t["track_id"])
        tracks[t["track_id"]] = {
            "state": t["state"],
            "angle": round(t["angle"], 12),
            "velocity": round(t["velocity_deg_per_scan"], 12),
            "history": [
                (p["version"], round(p["angle"], 12), p["kind"])
                for p in h["history"]
            ],
        }
    events = [
        (row["version"], row["events"])
        for row in manager.store.load_track_events(sid)
    ]
    results = [
        (
            r["version"],
            r.get("tracking", {}).get("angle_tracks")
            if r.get("tracking") else None,
        )
        for r in manager.store.load_results(sid)
    ]
    return tracks, events, results


def test_restart_matches_one_shot_run(tmp_path):
    """任中一点停服再起来接着推：航迹编号/历史/事件/归属与一口气跑完一致
    （时间戳除外）。"""
    from app.api.main import broker
    from app.sessions.session import SessionManager

    cfg = tracking_session_config()

    # 一口气跑完
    d1 = str(tmp_path / "one_shot")
    m1 = SessionManager(d1, broker)
    s1 = m1.create_session(copy.deepcopy(cfg))
    _feed_crossing(s1, 40)

    # 在第 20 批前“重启”：新 manager 指向同一目录
    d2 = str(tmp_path / "restart")
    m2 = SessionManager(d2, broker)
    s2 = m2.create_session(copy.deepcopy(cfg))
    sid2 = s2.id
    _feed_crossing(s2, 20)
    m2b = SessionManager(d2, broker)  # 模拟进程重启：仅从磁盘恢复
    s2b = m2b.get(sid2)
    for i in range(20, 40):
        th1 = -15.0 + 30.0 * i / 39.0
        th2 = 15.0 - 30.0 * i / 39.0
        real, imag = synthetic_batch(
            M, [th1, th2], [1.0, 0.5], 15.0, CROSS_SEED + i, k=K
        )
        s2b.append(f"b{i}", real, imag, run_estimate=True)

    a = _track_snapshot(m1, s1.id)
    b = _track_snapshot(m2b, sid2)
    assert a == b


def test_restart_then_duplicate_batch_no_change(tmp_path):
    """重启后已计入批次号仍幂等：重交不产生任何航迹事件、版本不变。"""
    from app.api.main import broker
    from app.sessions.session import SessionManager

    cfg = tracking_session_config()
    d = str(tmp_path / "restart_dup")
    m = SessionManager(d, broker)
    s = m.create_session(copy.deepcopy(cfg))
    sid = s.id
    _feed_crossing(s, 12)
    m2 = SessionManager(d, broker)
    s2 = m2.get(sid)
    version_before = s2.status()["version"]
    events_before = len(m2.store.load_track_events(sid))

    real, imag = synthetic_batch(
        M, [-15.0 + 30.0 * 5 / 39.0, 15.0 - 30.0 * 5 / 39.0],
        [1.0, 0.5], 15.0, CROSS_SEED + 5, k=K,
    )
    resp = s2.append("b5", real, imag, run_estimate=True)
    assert resp["duplicate"] is True
    assert s2.status()["version"] == version_before
    assert len(m2.store.load_track_events(sid)) == events_before


# ----------------------------------------------------------------------
def test_concurrent_appends_match_serial_replay(client):
    """12 批并发推送、各自触发估计：最终航迹 == 按服务实际处理顺序串行
    重放的结果；同一批次号再交一次不产生任何航迹事件。"""
    sid = _new_session(client, window_size=600)

    # 准备 12 批（运动目标，使处理顺序真正影响逐次关联）
    data = {}
    for i in range(12):
        a = [-12.0 + 1.5 * i, 12.0 - 1.5 * i]
        real, imag = synthetic_batch(M, a, [1.0, 0.6], 15.0, 7000 + i, k=K)
        data[f"c{i}"] = (real.tolist(), imag.tolist())

    errors: list[Exception] = []

    def worker(bid):
        try:
            r = client.post(f"/sessions/{sid}/append", json={
                "batch_id": bid, "real": data[bid][0], "imag": data[bid][1],
                "estimate": True,
            })
            assert r.status_code == 200, r.text
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(b,)) for b in sorted(data)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []

    # 服务实际处理顺序 = 结果版本号对应的 trigger_batch_id 顺序
    results = client.get(f"/sessions/{sid}/results",
                         params={"since_version": 0}).json()["results"]
    order = [r["trigger_batch_id"] for r in results]
    assert len(order) == 12
    # 同一次估计的每个角度都恰好归属一次（不重不漏）
    for r in results:
        angles = r["result"]["angles"]
        at = r["tracking"]["angle_tracks"]
        assert len(at) == len(angles)
        assoc = [x for x in at if x["associated"]]
        assert len({x["track_id"] for x in assoc}) == len(assoc)

    # 按实际顺序串行重放到另一个会话
    sid_ref = _new_session(client, window_size=600)
    for bid in order:
        r = client.post(f"/sessions/{sid_ref}/append", json={
            "batch_id": bid, "real": data[bid][0], "imag": data[bid][1],
            "estimate": True,
        })
        assert r.json()["duplicate"] is False

    def snapshot(s):
        tracks = {}
        for t in _tracks(client, s, include_terminated=True):
            h = _history(client, s, t["track_id"])
            tracks[t["track_id"]] = (
                t["state"], round(t["angle"], 12),
                [(p["version"], round(p["angle"], 12), p["kind"])
                 for p in h["history"]],
            )
        evs = [
            (row["version"], row["events"]) for row in _events(client, s)
        ]
        blocks = [
            (r["version"], r["tracking"]["angle_tracks"])
            for r in client.get(f"/sessions/{s}/results",
                                params={"since_version": 0}).json()["results"]
        ]
        return tracks, evs, blocks

    assert snapshot(sid) == snapshot(sid_ref)

    # 重复提交任一批次号：无版本、无航迹事件
    version_before = client.get(f"/sessions/{sid}").json()["version"]
    events_before = len(_events(client, sid))
    bid = order[0]
    r = client.post(f"/sessions/{sid}/append", json={
        "batch_id": bid, "real": data[bid][0], "imag": data[bid][1],
        "estimate": True,
    }).json()
    assert r["duplicate"] is True
    assert client.get(f"/sessions/{sid}").json()["version"] == version_before
    assert len(_events(client, sid)) == events_before


# ----------------------------------------------------------------------
def test_tracking_disabled_session_unchanged(client):
    """未开航迹的会话行为与旧版完全一致：估计信封没有 tracking 字段。"""
    from tests.conftest import create_session_config

    sid = client.post("/sessions", json=create_session_config()).json()["session_id"]
    real, imag = synthetic_batch(M, [-20.0, 25.0], [1.0, 1.0], 20.0, 99, k=64)
    body = client.post(f"/sessions/{sid}/append", json={
        "batch_id": "b0", "real": real.tolist(), "imag": imag.tolist(),
        "estimate": True,
    }).json()
    assert "tracking" not in body["result"]
    status = client.get(f"/sessions/{sid}").json()
    assert status["tracking_enabled"] is False
    # 航迹接口给出明确的人读错误
    r = client.get(f"/sessions/{sid}/tracks")
    assert r.status_code == 422
    assert r.json()["error"]["type"] == "validation_error"

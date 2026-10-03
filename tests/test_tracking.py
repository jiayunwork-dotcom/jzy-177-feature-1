"""航迹验收剧本：静止双目标 / 中途进出 / 一闪而过 / 交叉 / 重启 / 并发。

数据约定（与外场验收一致）：八阵元、半波长间距，每批 50 条快拍、
滑动窗口 50（每批协方差只由本批决定），每推一批出一次估计，种子固定。
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from app.doa.engine import ArraySpec
from app.doa.synthesis import generate_snapshots

M, SPACING, WL, K, WINDOW, STEP = 8, 0.5, 1.0, 50, 50, 0.2
SPEC = ArraySpec(elements=M, spacing=SPACING, wavelength=WL)
POSITIONS = SPEC.positions()


def tracking_config(**kw):
    cfg = {
        "array": {"elements": M, "spacing": SPACING, "wavelength": WL},
        "estimation": {
            "angle_min": -90.0, "angle_max": 90.0, "angle_step": STEP,
        },
        "update": {"mode": "sliding", "window_size": WINDOW},
        "tracking": {"enabled": True},
    }
    cfg.update(kw)
    return cfg


def batch(angles, powers, snr_db, seed):
    """造一批 50 条快拍（固定种子可复现）。"""
    syn = generate_snapshots(
        m=M, k=K, angles_deg=list(angles), powers=list(powers),
        snr_db=snr_db, positions=POSITIONS, wavelength=WL, seed=seed,
    )
    return syn["data"]


def push(client, sid, i, data, estimate=True):
    return client.post(f"/sessions/{sid}/append", json={
        "batch_id": f"batch-{i:03d}",
        "real": np.real(data).tolist(),
        "imag": np.imag(data).tolist(),
        "estimate": estimate,
    })


def create_tracking_session(client):
    return client.post("/sessions", json=tracking_config()).json()["session_id"]


def all_tracks(client, sid):
    return client.get(f"/sessions/{sid}/tracks").json()["tracks"]


def track_by_id(client, sid, tid):
    return client.get(f"/sessions/{sid}/tracks/{tid}").json()


def event_types(client, sid):
    ev = client.get(f"/sessions/{sid}/track-events",
                    params={"since_version": 0}).json()["events"]
    return [(e["version"], e["type"], e.get("track_id")) for e in ev]


def canonical_tracking_state(client, sid):
    """把一口气跑完的结果归一化成可逐项比较的结构（去掉时间戳）。"""
    tracks = []
    for t in all_tracks(client, sid):
        d = track_by_id(client, sid, t["track_id"])
        tracks.append({
            "track_id": d["track_id"],
            "status": d["status"],
            "hits": d["hits"],
            "first_version": d["first_version"],
            "last_hit_version": d["last_hit_version"],
            "history": [
                {"version": p["version"],
                 "angle": round(p["angle"], 12),
                 "power": round(p["power"], 12)}
                for p in d["history"]
            ],
        })
    results = client.get(f"/sessions/{sid}/results",
                         params={"since_version": 0}).json()["results"]
    envelopes = []
    for r in results:
        tr = r.get("tracking")
        assert tr is not None
        envelopes.append({
            "version": r["version"],
            "assignments": [
                {"angle_index": a["angle_index"],
                 "angle": round(a["angle"], 12),
                 "track_id": a["track_id"]}
                for a in tr["assignments"]
            ],
            "crossing": tr["crossing"],
            "events": [
                {k: v for k, v in e.items() if k != "created_at"}
                for e in tr["events"]
            ],
        })
    events = [
        {k: v for k, v in e.items() if k != "created_at"}
        for e in client.get(f"/sessions/{sid}/track-events",
                            params={"since_version": 0}).json()["events"]
    ]
    status = client.get(f"/sessions/{sid}").json()
    return {
        "version": status["version"],
        "n_tracks": status["n_tracks"],
        "tracks": sorted(tracks, key=lambda t: t["track_id"]),
        "envelopes": envelopes,
        "events": events,
    }


# ======================================================================
# 1) 静止双目标
# ======================================================================
def test_stationary_two_targets_two_stable_tracks(client):
    sid = create_tracking_session(client)
    for i in range(30):
        data = batch([-20.0, 25.0], [1.0, 1.0], 20.0, seed=100)
        r = push(client, sid, i, data)
        assert r.status_code == 200

    tracks = all_tracks(client, sid)
    confirmed = [t for t in tracks if t["status"] == "confirmed"]
    assert len(confirmed) == 2
    assert len(tracks) == 2  # 没有任何旁支航迹
    # 编号 30 批不变：每条航迹每个版本都在、且只归属自己
    t1 = track_by_id(client, sid, 1)
    t2 = track_by_id(client, sid, 2)
    assert [p["version"] for p in t1["history"]] == list(range(1, 31))
    assert [p["version"] for p in t2["history"]] == list(range(1, 31))
    # 角度与真值差在 0.1° 以内
    assert all(abs(p["angle"] - (-20.0)) < 0.1 for p in t1["history"])
    assert all(abs(p["angle"] - 25.0) < 0.1 for p in t2["history"])

    types = [t for _, t, _ in event_types(client, sid)]
    assert types.count("track_appeared") == 2
    assert types.count("track_confirmed") == 2
    assert not any(t in ("track_lost", "track_terminated",
                        "crossing_started") for t in types)


# ======================================================================
# 2) 中途进出
# ======================================================================
def test_appears_midway_then_disappears_lifecycle(client):
    sid = create_tracking_session(client)
    for i in range(30):
        present = 10 <= i <= 19
        angles = [-20.0, 25.0] + ([50.0] if present else [])
        powers = [1.0] * len(angles)
        push(client, sid, i, batch(angles, powers, 20.0, seed=3100 + i))

    t1 = track_by_id(client, sid, 1)
    t2 = track_by_id(client, sid, 2)
    t3 = track_by_id(client, sid, 3)

    # 前两条全程 confirmed、编号不变、角度对
    assert t1["status"] == t2["status"] == "confirmed"
    assert len(t1["history"]) == len(t2["history"]) == 30
    assert all(abs(p["angle"] + 20.0) < 0.2 for p in t1["history"])
    assert all(abs(p["angle"] - 25.0) < 0.2 for p in t2["history"])

    # 第三条完整经历 出现 -> 确认 -> 丢失 -> 终止
    assert t3["status"] == "terminated"
    ev = event_types(client, sid)
    seq = [(v, t) for v, t, tid in ev if tid == 3]
    versions = {t: v for v, t in seq}
    assert versions["track_appeared"] == 11
    assert versions["track_confirmed"] == 13
    assert versions["track_lost"] == 23
    assert versions["track_terminated"] == 26
    # 存在期间每拍都归属 50°
    assert [p["version"] for p in t3["history"]] == list(range(11, 21))
    assert all(abs(p["angle"] - 50.0) < 0.3 for p in t3["history"])


# ======================================================================
# 3) 一闪而过
# ======================================================================
def test_single_batch_blip_leaves_no_confirmed_track(client):
    sid = create_tracking_session(client)
    for i in range(12):
        present = i == 5
        angles = [-20.0, 25.0] + ([50.0] if present else [])
        push(client, sid, i, batch(angles, [1.0] * len(angles),
                                   20.0, seed=4000 + i))

    confirmed_ids = {t["track_id"] for t in all_tracks(client, sid)
                     if t["status"] == "confirmed"}
    assert confirmed_ids == {1, 2}
    # 假峰只留下一条从未确认、直接终止的候选航迹
    blip = track_by_id(client, sid, 3)
    assert blip["status"] == "terminated"
    assert blip["hits"] == 1
    assert blip["history"][0]["version"] == 6
    ev = [(t, tid) for _, t, tid in event_types(client, sid) if tid == 3]
    assert ev == [("track_appeared", 3), ("track_terminated", 3)]


# ======================================================================
# 4) 交叉
# ======================================================================
def test_crossing_targets_keep_identity(client):
    sid = create_tracking_session(client)
    true_traj = []
    for i in range(40):
        t = i / 39.0
        a1, a2 = -15.0 + 30.0 * t, 15.0 - 30.0 * t
        true_traj.append((a1, a2))
        push(client, sid, i, batch([a1, a2], [1.0, 0.5], 15.0, seed=2026))

    tracks = all_tracks(client, sid)
    by_id = {t["track_id"]: t for t in tracks}
    # 恰好两条、全程 confirmed，没有丢失/终止/合并出的新航迹
    assert set(by_id) == {1, 2}
    assert by_id[1]["status"] == by_id[2]["status"] == "confirmed"
    t1 = track_by_id(client, sid, 1)
    t2 = track_by_id(client, sid, 2)

    # 穿越之后（取后半程），T1 跟的是从 -15 走到 +15 的那个源
    late1 = [p for p in t1["history"] if p["version"] >= 30]
    late2 = [p for p in t2["history"] if p["version"] >= 30]
    assert late1 and late2
    assert all(p["angle"] > 0 for p in late1)
    assert all(p["angle"] < 0 for p in late2)
    # 穿越之前 T1 在负角、T2 在正角，编号没换
    early1 = [p for p in t1["history"] if p["version"] <= 15]
    early2 = [p for p in t2["history"] if p["version"] <= 15]
    assert all(p["angle"] < 0 for p in early1)
    assert all(p["angle"] > 0 for p in early2)
    # 后段逐点跟真值走（源 1：-15 -> +15）
    for p in late1:
        truth = true_traj[p["version"] - 1][0]
        assert abs(p["angle"] - truth) < 0.5
    for p in late2:
        truth = true_traj[p["version"] - 1][1]
        assert abs(p["angle"] - truth) < 0.5

    # 交叉期间两条航迹都没有被判丢失
    ev = event_types(client, sid)
    assert not any(t in ("track_lost", "track_terminated")
                   for _, t, _ in ev)
    crossing = [v for v, t, _ in ev if t == "crossing_started"]
    resolved = [v for v, t, _ in ev if t == "crossing_resolved"]
    assert crossing and resolved and crossing[0] < resolved[0]

    # 已推给上位机的归属不回头改：results 里每次归属落定即固定
    rows = client.get(f"/sessions/{sid}/results",
                      params={"since_version": 0}).json()["results"]
    assert len(rows) == 40


# ======================================================================
# 5) 重启：与一口气跑完逐项一致（时间戳除外）
# ======================================================================
def _restart_scenario_batches():
    batches = []
    for i in range(40):
        t = i / 39.0
        batches.append(batch([-15.0 + 30.0 * t, 15.0 - 30.0 * t],
                             [1.0, 0.5], 15.0, seed=2026))
    return batches


def test_restart_midway_matches_uninterrupted_run(client, tmp_path):
    from app.api.main import broker
    from app.sessions.session import SessionManager

    data_dir = str(tmp_path / "restart_tracks")

    # 一口气参考
    ref_mgr = SessionManager(data_dir + "-ref", broker)
    ref_client = _manager_client(ref_mgr)
    ref_sid = create_tracking_session(ref_client)
    batches = _restart_scenario_batches()
    for i, data in enumerate(batches):
        push(ref_client, ref_sid, i, data)
    ref = canonical_tracking_state(ref_client, ref_sid)

    # 跑到一半“停服”：丢弃 manager，同目录新建进程等价物后接着推
    mgr1 = SessionManager(data_dir, broker)
    c1 = _manager_client(mgr1)
    sid = create_tracking_session(c1)
    for i in range(20):
        push(c1, sid, i, batches[i])

    mgr2 = SessionManager(data_dir, broker)
    c2 = _manager_client(mgr2)
    for i in range(20, 40):
        push(c2, sid, i, batches[i])
    got = canonical_tracking_state(c2, sid)

    assert got == ref


def test_old_session_dir_without_tracking_files_still_works(tmp_path):
    """升级前已有的会话目录（无 track_events.jsonl / tracker 状态）照常读写。"""
    from app.api.main import broker
    from app.sessions.session import SessionManager

    data_dir = str(tmp_path / "old_sessions")
    mgr = SessionManager(data_dir, broker)
    cfg = tracking_config()
    cfg["tracking"] = {"enabled": False}
    s = mgr.create_session(cfg)
    assert not (mgr.store.session_dir(s.id) / "track_events.jsonl").exists()

    mgr2 = SessionManager(data_dir, broker)
    s2 = mgr2.get(s.id)
    assert s2.status()["tracking_enabled"] is False
    data = batch([-20.0, 25.0], [1.0, 1.0], 20.0, seed=100)
    env = s2.append("b0", data.real, data.imag, run_estimate=True)
    assert "tracking" not in env
    assert s2.status()["version"] == 1


def test_legacy_session_without_tracking_key_loads(tmp_path):
    """config.json 里根本没有 tracking 键（升级前落盘的样子）。"""
    import json

    from app.api.main import broker
    from app.sessions.session import SessionManager

    data_dir = str(tmp_path / "legacy")
    mgr = SessionManager(data_dir, broker)
    cfg = tracking_config()
    cfg.pop("tracking")
    s = mgr.create_session(cfg)  # 直接走存储建目录，config 无 tracking 键
    assert s.status()["tracking_enabled"] is False
    # 目录形态与旧版一致
    files = sorted(p.name for p in mgr.store.session_dir(s.id).iterdir())
    assert "track_events.jsonl" not in files

    mgr2 = SessionManager(data_dir, broker)
    s2 = mgr2.get(s.id)
    assert s2.status()["tracking_enabled"] is False
    data = batch([-20.0, 25.0], [1.0, 1.0], 20.0, seed=100)
    s2.append("b0", data.real, data.imag)
    assert s2.status()["n_accepted_batches"] == 1
    # state.json 仍不引入 tracker 段
    state = json.loads((mgr2.store.session_dir(s.id) / "state.json").read_text())
    assert "tracker" not in state


def test_restart_inside_crossing_keeps_identity(client, tmp_path):
    """交叉保持正中间停服重启：组状态也落盘，续写后编号依然不换。"""
    from app.api.main import broker
    from app.sessions.session import SessionManager

    data_dir = str(tmp_path / "restart_inside")
    batches = _restart_scenario_batches()

    ref_mgr = SessionManager(data_dir + "-ref", broker)
    ref_client = _manager_client(ref_mgr)
    ref_sid = create_tracking_session(ref_client)
    for i, data in enumerate(batches):
        push(ref_client, ref_sid, i, data)
    ref = canonical_tracking_state(ref_client, ref_sid)

    # 在 MDL 少报的 b=19..20 之间停（先推 19 批），再起来接着推
    mgr1 = SessionManager(data_dir, broker)
    c1 = _manager_client(mgr1)
    sid = create_tracking_session(c1)
    for i in range(19):
        push(c1, sid, i, batches[i])
    mid = client_get_status(c1, sid)
    # 此刻确有正在保持的交叉组
    rows1 = c1.get(f"/sessions/{sid}/results",
                   params={"since_version": 0}).json()["results"]
    assert any(r["tracking"]["crossing"]["active"] for r in rows1[-3:])

    mgr2 = SessionManager(data_dir, broker)
    c2 = _manager_client(mgr2)
    for i in range(19, 40):
        push(c2, sid, i, batches[i])
    got = canonical_tracking_state(c2, sid)
    assert got == ref


def client_get_status(client, sid):
    return client.get(f"/sessions/{sid}").json()


# ======================================================================
# 6) 并发：12 批并发推送，结果等价于按实际处理顺序串行重放；重复无事件
# ======================================================================
def test_concurrent_appends_tracks_match_serial_replay(client):
    sid = create_tracking_session(client)
    rng = np.random.default_rng(20260)
    # 两个静止源；批间噪声独立（不同种子），顺序不同 -> 窗口协方差不同
    batches = {}
    for i in range(12):
        batches[f"batch-{i:03d}"] = batch(
            [-20.0, 25.0], [1.0, 1.0], 20.0, seed=7000 + i)

    errors = []

    def worker(bid, data):
        try:
            r = client.post(f"/sessions/{sid}/append", json={
                "batch_id": bid,
                "real": np.real(data).tolist(),
                "imag": np.imag(data).tolist(),
                "estimate": True,
            })
            assert r.status_code == 200
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(b, d))
               for b, d in batches.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []

    rows = client.get(f"/sessions/{sid}/results",
                      params={"since_version": 0}).json()["results"]
    order = [r["trigger_batch_id"] for r in rows]
    assert sorted(order) == sorted(batches)
    assert len(order) == 12  # 同一次估计只关联一回，不漏不重

    # 按服务实际处理顺序串行重放到另一个会话，逐项比较
    sid2 = create_tracking_session(client)
    for n, bid in enumerate(order):
        push(client, sid2, n, batches[bid])
    a = canonical_tracking_state(client, sid)
    b_state = canonical_tracking_state(client, sid2)
    # 串行重放的批次号是重编的，trigger 批次号不可比；航迹内容应逐项一致
    assert a["n_tracks"] == b_state["n_tracks"]
    assert len(a["tracks"]) == len(b_state["tracks"])
    for ta, tb in zip(a["tracks"], b_state["tracks"]):
        assert ta["status"] == tb["status"]
        assert ta["history"] == tb["history"]
    for ea, eb in zip(a["envelopes"], b_state["envelopes"]):
        assert ea["assignments"] == eb["assignments"]
        assert ea["crossing"] == eb["crossing"]
        assert ea["events"] == eb["events"]

    # 同一批次号再交一次：不产生任何航迹变化
    before = client.get(f"/sessions/{sid}/track-events",
                        params={"since_version": 0}).json()
    dup = client.post(f"/sessions/{sid}/append", json={
        "batch_id": order[0],
        "real": np.real(batches[order[0]]).tolist(),
        "imag": np.imag(batches[order[0]]).tolist(),
        "estimate": True,
    }).json()
    assert dup["duplicate"] is True
    assert "result" not in dup
    after = client.get(f"/sessions/{sid}/track-events",
                       params={"since_version": 0}).json()
    assert before["events"] == after["events"]
    status = client.get(f"/sessions/{sid}").json()
    assert status["n_accepted_batches"] == 12
    assert status["version"] == 12


def test_track_events_polling_no_loss_no_duplicate(client):
    sid = create_tracking_session(client)
    for i in range(30):
        present = 10 <= i <= 19
        angles = [-20.0, 25.0] + ([50.0] if present else [])
        push(client, sid, i, batch(angles, [1.0] * len(angles),
                                   20.0, seed=3100 + i))
    # 分页轮询：不漏不重，版本严格递增
    seen = []
    since = 0
    for _ in range(100):
        page = client.get(f"/sessions/{sid}/track-events",
                          params={"since_version": since}).json()["events"]
        if not page:
            break
        assert all(e["version"] > since for e in page)
        seen.extend(page)
        since = page[-1]["version"]
    full = client.get(f"/sessions/{sid}/track-events",
                      params={"since_version": 0}).json()["events"]
    assert seen == full


# ----------------------------------------------------------------------
def _manager_client(manager):
    """给独立 SessionManager 套一个 TestClient（替换全局 manager）。"""
    from fastapi.testclient import TestClient
    import app.api.main as main_mod

    main_mod.manager = manager
    return TestClient(main_mod.app)

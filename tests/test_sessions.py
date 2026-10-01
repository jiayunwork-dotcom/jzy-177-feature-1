"""会话层与 HTTP：追加/幂等、持久化重启、轮询、SSE、并发隔离、错误码。"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest

from tests.conftest import create_session_config, make_data


def _append(client, sid, batch_id, data, estimate=False, **kw):
    payload = {
        "batch_id": batch_id,
        "real": np.real(data).tolist(),
        "imag": np.imag(data).tolist(),
        "estimate": estimate,
    }
    payload.update(kw)
    return client.post(f"/sessions/{sid}/append", json=payload)


def _create(client, body=None):
    return client.post("/sessions", json=body or create_session_config())


def test_create_and_status(client):
    r = _create(client)
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 0
    assert body["active_snapshots"] == 0


def test_append_estimate_poll_flow(client):
    sid = _create(client).json()["session_id"]
    _, syn = make_data(k=64)
    r = _append(client, sid, "b1", syn["data"], estimate=True)
    assert r.status_code == 200
    body = r.json()
    assert body["duplicate"] is False
    assert body["active_snapshots"] == 64
    assert "result" in body
    angles = body["result"]["result"]["angles"]
    assert pytest.approx(angles, abs=0.2) == sorted([-25.0, 25.0])

    # 轮询：since=0 拿到 v1；since=1 为空
    p1 = client.get(f"/sessions/{sid}/results", params={"since_version": 0}).json()
    assert len(p1["results"]) == 1 and p1["results"][0]["version"] == 1
    p2 = client.get(f"/sessions/{sid}/results", params={"since_version": 1}).json()
    assert p2["results"] == []

    # 再来一次估计 -> v2
    r2 = client.post(f"/sessions/{sid}/estimate", json={})
    assert r2.json()["version"] == 2
    p3 = client.get(f"/sessions/{sid}/results", params={"since_version": 1}).json()
    assert [x["version"] for x in p3["results"]] == [2]


def test_duplicate_batch_id_counted_once(client):
    sid = _create(client).json()["session_id"]
    _, syn = make_data(k=32)
    r1 = _append(client, sid, "dup-1", syn["data"])
    r2 = _append(client, sid, "dup-1", syn["data"])
    assert r1.json()["duplicate"] is False
    assert r2.json()["duplicate"] is True
    assert r2.json()["active_snapshots"] == 32  # 没有重复计入


def test_concurrent_appends_equivalent_to_serial(client):
    """同一会话并发多批：等价某顺序串行追加，不丢批不重复。"""
    import threading

    sid = _create(client).json()["session_id"]
    rng = np.random.default_rng(7)
    batches = {f"c{i}": (rng.standard_normal((8, 17)) +
                         1j * rng.standard_normal((8, 17))) / np.sqrt(2)
               for i in range(12)}
    errors = []

    def worker(bid, data):
        try:
            resp = _append(client, sid, bid, data)
            assert resp.status_code == 200
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(b, d))
               for b, d in batches.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    status = client.get(f"/sessions/{sid}").json()
    assert status["active_snapshots"] == 12 * 17
    assert status["n_accepted_batches"] == 12

    # 串行参考：另建会话，同样 12 批顺序追加，协方差应一致
    sid2 = _create(client).json()["session_id"]
    for bid in sorted(batches):
        _append(client, sid2, bid, batches[bid])
    # 两会话窗口未满（204 < 500），样本协方差与顺序无关
    r_a = client.post(f"/sessions/{sid}/estimate", json={}).json()["result"]
    r_b = client.post(f"/sessions/{sid2}/estimate", json={}).json()["result"]
    # 特征值（进而 MDL/MUSIC）与顺序无关
    assert r_a["eigenvalues"] == pytest.approx(r_b["eigenvalues"], rel=1e-10, abs=1e-12)


def test_sessions_isolated(client):
    s1 = _create(client).json()["session_id"]
    s2 = _create(client).json()["session_id"]
    _, d1 = make_data(angles=(-30.0,), k=64, seed=1)
    _, d2 = make_data(angles=(45.0,), k=64, seed=2)
    _append(client, s1, "x", d1["data"], estimate=True)
    _append(client, s2, "y", d2["data"], estimate=True)
    a1 = client.get(f"/sessions/{s1}/results/latest").json()["result"]["result"]["angles"]
    a2 = client.get(f"/sessions/{s2}/results/latest").json()["result"]["result"]["angles"]
    assert a1 == pytest.approx([-30.0], abs=0.2)
    assert a2 == pytest.approx([45.0], abs=0.2)
    # 版本号各自独立
    assert client.get(f"/sessions/{s1}").json()["version"] == 1
    assert client.get(f"/sessions/{s2}").json()["version"] == 1


def test_persistence_survives_restart(tmp_path):
    """重启后会话可接着追加，历史结果不丢（新 manager 指向同一目录）。"""
    from app.api.main import broker
    from app.sessions.session import SessionManager

    data_dir = str(tmp_path / "persist")
    mgr1 = SessionManager(data_dir, broker)
    cfg = create_session_config()
    s = mgr1.create_session(cfg)
    _, syn = make_data(k=128)
    s.append("b1", syn["data"][:].real, syn["data"][:].imag)
    s.estimate()
    v1 = s._version

    # 模拟重启：全新 manager / Session 对象，仅从磁盘恢复
    mgr2 = SessionManager(data_dir, broker)
    s2 = mgr2.get(s.id)
    assert s2.status()["version"] == v1
    assert s2.status()["active_snapshots"] == 128
    old = mgr2.store.load_results(s.id)
    assert len(old) == 1
    # 接着追加，版本延续
    _, more = make_data(k=64, seed=999)
    s2.append("b2", more["data"].real, more["data"].imag)
    assert s2.status()["active_snapshots"] == 128 + 64
    env = s2.estimate()
    assert env["version"] == v1 + 1
    assert len(mgr2.store.load_results(s.id)) == 2


def test_persistence_forgetting_state_roundtrip(tmp_path):
    from app.api.main import broker
    from app.sessions.session import SessionManager

    data_dir = str(tmp_path / "persist_f")
    mgr = SessionManager(data_dir, broker)
    cfg = create_session_config()
    cfg["update"] = {"mode": "forgetting", "forgetting_factor": 0.96}
    s = mgr.create_session(cfg)
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((8, 300)) + 1j * rng.standard_normal((8, 300))) / np.sqrt(2)
    s.append("f1", x[:, :100].real, x[:, :100].imag)
    s.append("f2", x[:, 100:].real, x[:, 100:].imag)
    r_before = s._acc.covariance()

    mgr2 = SessionManager(data_dir, broker)
    s2 = mgr2.get(s.id)
    assert np.allclose(s2._acc.covariance(), r_before, rtol=1e-12)
    # 重启后继续追加，批次号去重记录仍在
    dup = s2.append("f1", x[:, :100].real, x[:, :100].imag)
    assert dup["duplicate"] is True
    assert s2.status()["active_snapshots"] == 300


def test_whole_batch_equals_chunked_appends(client):
    """同样快拍整批一次送入与切几批陆续追加，最终估计一致。"""
    _, syn = make_data(angles=(-18.0, 22.0), snr_db=25.0, k=300, seed=11)
    cfg = create_session_config()
    cfg["update"] = {"mode": "forgetting", "forgetting_factor": 1.0}

    sid_a = client.post("/sessions", json=cfg).json()["session_id"]
    _append(client, sid_a, "all", syn["data"], estimate=True)
    sid_b = client.post("/sessions", json=cfg).json()["session_id"]
    sizes = [37, 88, 50, 125]
    idx = 0
    for i, n in enumerate(sizes):
        _append(client, sid_b, f"p{i}", syn["data"][:, idx:idx + n])
        idx += n
    client.post(f"/sessions/{sid_b}/estimate", json={})

    ra = client.get(f"/sessions/{sid_a}/results/latest").json()["result"]["result"]
    rb = client.get(f"/sessions/{sid_b}/results/latest").json()["result"]["result"]
    assert rb["angles"] == pytest.approx(ra["angles"], abs=1e-8)


def test_persistence_sliding_overflow_roundtrip(tmp_path):
    """窗口溢出、环形缓冲回绕之后重启：协方差仍等于批处理窗口定义。"""
    from app.api.main import broker
    from app.sessions.accumulator import batch_sliding_covariance
    from app.sessions.session import SessionManager

    data_dir = str(tmp_path / "persist_overflow")
    mgr = SessionManager(data_dir, broker)
    cfg = create_session_config()
    cfg["update"] = {"mode": "sliding", "window_size": 64}
    s = mgr.create_session(cfg)
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((8, 200)) + 1j * rng.standard_normal((8, 200))) / np.sqrt(2)
    # 大小不等的多批，跨越多次窗口回绕
    for i, (a, b) in enumerate([(0, 30), (30, 90), (90, 150), (150, 200)]):
        s.append(f"ov{i}", x[:, a:b].real, x[:, a:b].imag)

    mgr2 = SessionManager(data_dir, broker)
    s2 = mgr2.get(s.id)
    r_ref, n_ref = batch_sliding_covariance(x, 64)
    assert s2.status()["active_snapshots"] == n_ref == 64
    assert np.allclose(s2._acc.covariance(), r_ref, rtol=1e-12)
    # 已见批次号也恢复，重复整批不计入
    dup = s2.append("ov1", x[:, 30:90].real, x[:, 30:90].imag)
    assert dup["duplicate"] is True
    assert s2.status()["active_snapshots"] == 64


def test_sse_subscription_receives_new_results():
    """实时推送用真实 uvicorn + httpx 测：TestClient 的单事件循环
    无法同时承载流式连接与触发请求（会自锁）。"""
    import contextlib
    import threading
    import time

    import httpx
    import uvicorn

    import app.api.main as main_mod
    from app.api.main import app, broker
    from app.sessions.session import SessionManager

    tmp = tempfile.mkdtemp()
    mgr = SessionManager(tmp, broker)
    old = main_mod.manager
    main_mod.manager = mgr

    config = uvicorn.Config(app, host="127.0.0.1", port=8991, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _, syn = make_data(k=64)
    try:
        # 等待服务就绪
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        base = "http://127.0.0.1:8991"
        sid = httpx.post(f"{base}/sessions",
                         json=create_session_config()).json()["session_id"]
        with httpx.stream("GET", f"{base}/sessions/{sid}/stream",
                          timeout=15.0) as resp:
            assert resp.status_code == 200
            time.sleep(0.3)  # 确保流先建立订阅
            r = httpx.post(
                f"{base}/sessions/{sid}/append",
                json={"batch_id": "live",
                      "real": syn["data"].real.tolist(),
                      "imag": syn["data"].imag.tolist(),
                      "estimate": True},
                timeout=15.0,
            )
            assert r.status_code == 200
            payload = None
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    payload = json.loads(line[len("data:"):].strip())
                    break
        assert payload is not None and payload["version"] == 1
    finally:
        main_mod.manager = old
        server.should_exit = True
        thread.join(timeout=5)


def test_sse_replays_history_after_version(client):
    sid = _create(client).json()["session_id"]
    _, syn = make_data(k=64)
    _append(client, sid, "b1", syn["data"], estimate=True)
    # 历史补发在连接建立时立即发送；max_idle_seconds 让无新消息时流正常结束
    with client.stream(
        "GET", f"/sessions/{sid}/stream",
        params={"after_version": 0, "max_idle_seconds": 1.0},
    ) as resp:
        assert resp.status_code == 200
        payload = None
        for line in resp.iter_lines():
            if line.startswith("data:"):
                payload = json.loads(line[len("data:"):].strip())
                break
    assert payload is not None and payload["version"] == 1


# ---- HTTP 错误 ----------------------------------------------------------
def _check_error(resp, code=422, fragment=None):
    assert resp.status_code == code, resp.text
    body = resp.json()
    assert body["error"]["message"]
    if fragment:
        assert fragment in body["error"]["message"]
    return body


def test_http_nan_error(client):
    sid = _create(client).json()["session_id"]
    real = [[0.0] * 10 for _ in range(8)]
    imag = [[0.0] * 10 for _ in range(8)]
    imag[2][3] = float("nan")
    # httpx 默认拒绝非标准 JSON（NaN），这里用原始内容直发，
    # 由服务端 JSON 解析后内核报出人读 NaN 错误。
    payload = json.dumps({"batch_id": "n", "real": real, "imag": imag},
                         allow_nan=True)
    r = client.post(
        f"/sessions/{sid}/append",
        content=payload,
        headers={"Content-Type": "application/json"},
    )
    _check_error(r, fragment="NaN")


def test_http_shape_mismatch(client):
    sid = _create(client).json()["session_id"]
    r = client.post(f"/sessions/{sid}/append", json={
        "batch_id": "s",
        "real": [[0.0] * 10 for _ in range(8)],
        "imag": [[0.0] * 9 for _ in range(8)],
    })
    _check_error(r, fragment="形状不一致")


def test_http_estimate_too_few_snapshots(client):
    sid = _create(client).json()["session_id"]
    _, syn = make_data(k=3)  # K < M=8
    r = client.post(f"/sessions/{sid}/estimate")  # 0 快拍
    _check_error(r)
    _append(client, sid, "tiny", syn["data"])
    r2 = client.post(f"/sessions/{sid}/estimate", json={})
    _check_error(r2, fragment="少于阵元数")


def test_http_create_bad_wavelength(client):
    cfg = create_session_config()
    cfg["array"]["wavelength"] = 0
    r = client.post("/sessions", json=cfg)
    assert r.status_code == 422


def test_http_elements_limit(client):
    cfg = create_session_config(m=65)
    r = client.post("/sessions", json=cfg)
    assert r.status_code == 422


def test_http_batch_size_limit(client):
    sid = _create(client).json()["session_id"]
    r = client.post(f"/sessions/{sid}/append", json={
        "batch_id": "big",
        "real": [[0.0] * 10_001 for _ in range(8)],
        "imag": [[0.0] * 10_001 for _ in range(8)],
    })
    _check_error(r, fragment="10000")


def test_http_scan_points_limit(client):
    sid = _create(client).json()["session_id"]
    _, syn = make_data(k=64)
    _append(client, sid, "b", syn["data"])
    r = client.post(f"/sessions/{sid}/estimate",
                    json={"angle_step": 0.0005})  # 180/0.0005+1 = 360001
    _check_error(r, fragment="扫描点数")


def test_http_source_count_ge_elements(client):
    sid = _create(client).json()["session_id"]
    _, syn = make_data(k=64)
    _append(client, sid, "b", syn["data"])
    r = client.post(f"/sessions/{sid}/estimate", json={"source_count": 8})
    _check_error(r, fragment="不小于")


def test_http_session_not_found(client):
    r = client.get("/sessions/does-not-exist")
    assert r.status_code == 422
    assert r.json()["error"]["type"] == "session_not_found"


def test_session_without_estimation_config_uses_defaults(client):
    body = {"array": {"elements": 8, "spacing": 0.5, "wavelength": 1.0}}
    sid = client.post("/sessions", json=body).json()["session_id"]
    _, syn = make_data(k=64)
    _append(client, sid, "b", syn["data"])
    r = client.post(f"/sessions/{sid}/estimate",
                    json={"source_count": 2, "true_angles": [-25.0, 25.0]})
    assert r.status_code == 200
    assert all(abs(e) < 0.2 for e in r.json()["result"]["angle_error_deg"])


def test_forgetting_effective_snapshots_saturates():
    from app.sessions.accumulator import CovarianceAccumulator

    acc = CovarianceAccumulator(8, "forgetting", forgetting_factor=0.98)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((8, 10000)) + 1j * rng.standard_normal((8, 10000))) / np.sqrt(2)
    acc.add_batch(x[:, :200])
    n_small = acc.effective_snapshots
    acc.add_batch(x[:, 200:])
    n_large = acc.effective_snapshots
    # 稳态有效样本量 (1+λ)/(1-λ) = 99，不会随累计条数无限增长
    assert n_small < n_large <= 105
    assert n_large == 99


def test_estimate_with_true_angles_returns_error_over_crb(client):
    sid = _create(client).json()["session_id"]
    _, syn = make_data(k=128)
    _append(client, sid, "b", syn["data"])
    r = client.post(f"/sessions/{sid}/estimate",
                    json={"true_angles": [-25.0, 25.0]})
    assert r.status_code == 200
    res = r.json()["result"]
    assert res["angle_error_deg"] is not None
    assert all(x is not None for x in res["error_over_crb"])
    assert all(abs(e) < 0.1 for e in res["angle_error_deg"])


def test_synthesis_endpoint(client):
    body = {
        "elements": 8, "spacing": 0.5, "wavelength": 1.0,
        "k_snapshots": 400, "snr_db": 25.0, "seed": 42,
        "sources": [{"angle": -25, "power": 1}, {"angle": 25, "power": 1}],
        "angle_step": 0.5,
    }
    r = client.post("/synthesis", json=body)
    assert r.status_code == 200
    res = r.json()
    assert res["synthesis"]["true_angles"] == [-25.0, 25.0]
    assert res["result"]["angles"] == pytest.approx([-25.0, 25.0], abs=0.1)
    assert "noise_variance" in res["synthesis"]


def test_synthesis_endpoint_bit_identical_then_data_roundtrip(client):
    body = {
        "elements": 8, "spacing": 0.5, "wavelength": 1.0,
        "k_snapshots": 100, "snr_db": 15.0, "seed": 777,
        "sources": [{"angle": 5, "power": 1}],
        "estimate": False, "include_data": True,
    }
    a = client.post("/synthesis", json=body).json()
    b = client.post("/synthesis", json=body).json()
    assert a["data"]["real"] == b["data"]["real"]
    assert a["data"]["imag"] == b["data"]["imag"]
    # 回传数据可以原样喂进新建会话
    sid = _create(client).json()["session_id"]
    r = client.post(f"/sessions/{sid}/append", json={
        "batch_id": "syn", "real": a["data"]["real"], "imag": a["data"]["imag"],
    })
    assert r.status_code == 200

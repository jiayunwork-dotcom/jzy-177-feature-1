"""增量协方差：滑动窗口 / 遗忘因子与批处理定义的数值一致性。"""

from __future__ import annotations

import numpy as np
import pytest

from app.sessions.accumulator import (
    CovarianceAccumulator,
    batch_forgetting_covariance,
    batch_sliding_covariance,
)


@pytest.fixture
def rng():
    return np.random.default_rng(2026)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def test_sliding_matches_batch_definition_irregular_chunks(rng):
    m, k, w = 8, 300, 128
    x = (rng.standard_normal((m, k)) + 1j * rng.standard_normal((m, k))) / np.sqrt(2)
    acc = CovarianceAccumulator(m, "sliding", window_size=w)
    cuts = sorted(rng.choice(range(1, k), size=11, replace=False))
    start = 0
    for cut in list(cuts) + [k]:
        acc.add_batch(x[:, start:cut])
        start = cut
    r_ref, n_ref = batch_sliding_covariance(x, w)
    assert acc.n_snapshots == n_ref == w
    assert _rel(acc.covariance(), r_ref) < 1e-9


def test_sliding_before_window_full(rng):
    m, k, w = 8, 50, 200
    x = (rng.standard_normal((m, k)) + 1j * rng.standard_normal((m, k))) / np.sqrt(2)
    acc = CovarianceAccumulator(m, "sliding", window_size=w)
    acc.add_batch(x[:, :13])
    acc.add_batch(x[:, 13:])
    r_ref, _ = batch_sliding_covariance(x, w)
    assert _rel(acc.covariance(), r_ref) < 1e-12
    assert acc.n_snapshots == 50


def test_sliding_no_drift_after_many_appends(rng):
    """追加远超窗口长度后仍与批处理定义一致，且严格保持厄米性。"""
    m, k, w = 8, 6000, 256
    x = (rng.standard_normal((m, k)) + 1j * rng.standard_normal((m, k))) / np.sqrt(2)
    acc = CovarianceAccumulator(m, "sliding", window_size=w)
    # 100 快拍一批，共 60 批
    for i in range(0, k, 100):
        acc.add_batch(x[:, i:i + 100])
    r_ref, _ = batch_sliding_covariance(x, w)
    assert _rel(acc.covariance(), r_ref) < 1e-9
    r = acc.covariance()
    assert np.linalg.norm(r - r.conj().T) == 0.0


def test_forgetting_matches_batch_definition(rng):
    m, k = 8, 400
    lam = 0.95
    x = (rng.standard_normal((m, k)) + 1j * rng.standard_normal((m, k))) / np.sqrt(2)
    acc = CovarianceAccumulator(m, "forgetting", forgetting_factor=lam)
    sizes = [1, 7, 33, 128, 200, 31]
    start = 0
    for s in sizes:
        acc.add_batch(x[:, start:start + s])
        start += s
    r_ref, _ = batch_forgetting_covariance(x, lam)
    assert _rel(acc.covariance(), r_ref) < 1e-9


def test_forgetting_chunked_equals_single_batch(rng):
    """同批快拍一次送入与切成多批陆续追加，协方差严格一致。"""
    m, k, lam = 8, 250, 0.97
    x = (rng.standard_normal((m, k)) + 1j * rng.standard_normal((m, k))) / np.sqrt(2)
    a = CovarianceAccumulator(m, "forgetting", forgetting_factor=lam)
    a.add_batch(x)
    b = CovarianceAccumulator(m, "forgetting", forgetting_factor=lam)
    for i in range(0, k, 37):
        b.add_batch(x[:, i:i + 37])
    assert _rel(a.covariance(), b.covariance()) < 1e-14  # 同一更新次序，仅有舍入差


def test_forgetting_hermitian_long_run(rng):
    m, lam = 8, 0.99
    x = (rng.standard_normal((m, 5000)) + 1j * rng.standard_normal((m, 5000))) / np.sqrt(2)
    acc = CovarianceAccumulator(m, "forgetting", forgetting_factor=lam)
    acc.add_batch(x)
    r = acc.covariance()
    assert np.linalg.norm(r - r.conj().T) == 0.0
    # 与批处理参考一致（截断几何权重）
    r_ref, _ = batch_forgetting_covariance(x, lam)
    assert _rel(r, r_ref) < 1e-9


def test_invalid_parameters():
    with pytest.raises(Exception):
        CovarianceAccumulator(8, mode="bogus")
    with pytest.raises(Exception):
        CovarianceAccumulator(8, "forgetting", forgetting_factor=0.0)
    with pytest.raises(Exception):
        CovarianceAccumulator(8, "forgetting", forgetting_factor=1.5)
    with pytest.raises(Exception):
        CovarianceAccumulator(8, "sliding", window_size=0)

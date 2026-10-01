"""pytest 公共夹具：临时数据目录 + FastAPI TestClient + 合成辅助。"""

from __future__ import annotations

import numpy as np
import pytest

from app.doa.engine import ArraySpec, EstConfig
from app.doa.synthesis import generate_snapshots


@pytest.fixture
def tmp_data_dir(tmp_path):
    return str(tmp_path / "sessions_data")


@pytest.fixture
def client(tmp_data_dir):
    # 每个测试独立的 SessionManager / 数据目录
    from fastapi.testclient import TestClient

    from app.api.main import app, broker
    from app.sessions.session import SessionManager

    import app.api.main as main_mod

    mgr = SessionManager(tmp_data_dir, broker)
    old = main_mod.manager
    main_mod.manager = mgr
    with TestClient(app) as c:
        yield c
    main_mod.manager = old


def make_data(m=8, angles=(-25.0, 25.0), powers=None,
              snr_db=25.0, k=500, seed=42, spacing=0.5, wavelength=1.0,
              coherent=False, position_errors=None):
    if powers is None:
        powers = (1.0,) * len(angles)
    spec = ArraySpec(
        elements=m, spacing=spacing, wavelength=wavelength,
        position_errors=tuple(position_errors) if position_errors is not None else None,
    )
    syn = generate_snapshots(
        m=m, k=k, angles_deg=list(angles), powers=list(powers),
        snr_db=snr_db, positions=spec.positions(), wavelength=wavelength,
        seed=seed, coherent=coherent,
    )
    return spec, syn


def create_session_config(m=8, spacing=0.5, wavelength=1.0, **est_overrides):
    estimation = {"angle_min": -90.0, "angle_max": 90.0, "angle_step": 0.5}
    estimation.update(est_overrides)
    return {
        "array": {"elements": m, "spacing": spacing, "wavelength": wavelength},
        "estimation": estimation,
        "update": {"mode": "sliding", "window_size": 500},
    }


def chunks(x: np.ndarray, sizes):
    idx = 0
    for s in sizes:
        yield x[:, idx: idx + s]
        idx += s

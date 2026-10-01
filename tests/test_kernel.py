"""测向内核验收用例（直接调内核，不经过 HTTP）。"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from app.doa.array import grating_alias_pairs
from app.doa.engine import ArraySpec, EstConfig, estimate
from app.doa.synthesis import generate_snapshots
from app.errors import SourceCountError, ValidationError
from tests.conftest import make_data


def test_two_incoherent_sources_bias_under_0p1():
    """高 SNR、快拍够多、角度拉开的两个非相干源，各自偏差 < 0.1°。"""
    spec, syn = make_data(angles=(-28.0, 33.0), snr_db=25.0, k=800, seed=2024)
    res = estimate(syn["data"], spec, EstConfig(angle_step=0.5),
                   true_angles=syn["angles"])
    assert res["mdl_source_count"] == 2
    assert len(res["angles"]) == 2
    errs = [abs(e) for e in res["angle_error_deg"]]
    assert errs[0] < 0.1 and errs[1] < 0.1


def test_global_complex_scaling_leaves_angles_unchanged():
    """整批快拍同乘任意非零复数，估计角度逐位不变（协方差仅差标量）。"""
    spec, syn = make_data(angles=(-10.0, 40.0), snr_db=20.0, k=400, seed=5)
    cfg = EstConfig(angle_step=0.5)
    r1 = estimate(syn["data"], spec, cfg)
    c = (-4.7 + 2.3j) * cmath.exp(1j * 2.1)
    r2 = estimate(c * syn["data"], spec, cfg)
    assert r1["angles"] == pytest.approx(r2["angles"], abs=1e-12)


@pytest.mark.parametrize("spacing", [0.75, 1.0, 2.0])
def test_grating_lobe_ambiguity_reported_with_pairs(spacing):
    """间距 > 半波长：报告栅瓣，并给出会被混淆的角度对（sin 周期 lambda/d）。"""
    spec, syn = make_data(angles=(10.0, 40.0), spacing=spacing, snr_db=30.0, k=400)
    res = estimate(syn["data"], spec, EstConfig(angle_step=1.0))
    assert res["grating_ambiguity"]["present"] is True
    pairs = [tuple(p) for p in res["grating_ambiguity"]["alias_pairs"]]
    assert len(pairs) > 0
    wl = 1.0
    # 抽查：枚举出来的每对角度必须满足 sin(theta')-sin(theta)=k*lambda/d
    for a, b in pairs:
        delta = abs(math.sin(math.radians(a)) - math.sin(math.radians(b)))
        k_round = round(delta / (wl / spacing))
        assert k_round >= 1
        assert delta == pytest.approx(k_round * wl / spacing, abs=1e-6)


def test_no_grating_when_half_wavelength():
    spec, syn = make_data(spacing=0.5)
    res = estimate(syn["data"], spec, EstConfig(angle_step=1.0))
    assert res["grating_ambiguity"]["present"] is False
    assert res["grating_ambiguity"]["alias_pairs"] == []


def test_angle_sign_flip():
    """信源角度统一变号（等价于数据共轭），估计结果统一变号。"""
    spec, syn = make_data(angles=(-20.0, 15.0), snr_db=25.0, k=600, seed=99)
    cfg = EstConfig(angle_step=0.5)
    r_pos = estimate(syn["data"], spec, cfg)
    # 信源角度统一变号 = 数据整体取共轭（含噪声），这里直接构造同噪声
    # 实现的共轭数据；另用独立重抽（同种子、相反角度）复核统计意义变号。
    r_conj = estimate(np.conj(syn["data"]), spec, cfg)
    assert sorted(r_conj["angles"]) == pytest.approx(
        sorted(-a for a in r_pos["angles"]), abs=1e-9)

    spec2, syn2 = make_data(angles=(20.0, -15.0), snr_db=25.0, k=600, seed=99)
    r_neg = estimate(syn2["data"], spec2, cfg)
    assert sorted(r_neg["angles"]) == pytest.approx(
        sorted(-a for a in r_pos["angles"]), abs=0.1)


def test_coherent_sources_require_smoothing():
    """一对完全相干源：不开平滑分不开；开前后向平滑后分开。"""
    spec, syn = make_data(angles=(-12.5, 12.5), snr_db=30.0, k=800,
                          seed=7, coherent=True)
    # 不开平滑：两个估计角都无法同时落在真实方向附近（谱被秩亏破坏）
    r_no = estimate(syn["data"], spec,
                    EstConfig(angle_step=0.2, source_count=2),
                    true_angles=syn["angles"])
    assert not all(abs(e) < 1.0 for e in r_no["angle_error_deg"])
    # 开平滑：两个角各自偏差 < 0.5°
    r_fb = estimate(syn["data"], spec,
                    EstConfig(angle_step=0.2, smoothing=True, source_count=2),
                    true_angles=syn["angles"])
    assert len(r_fb["angles"]) == 2
    assert all(abs(e) < 0.5 for e in r_fb["angle_error_deg"])


def test_smoothing_exceeds_resolvable_sources_is_rejected():
    """信源数超过子阵配置的上限：拒绝并给出可读上限。"""
    spec, syn = make_data(angles=(0.0,), k=64)
    # M=8, L=3 => J=6, dmax=min(2,12)=2；请求 3 必须拒
    with pytest.raises(SourceCountError) as ei:
        estimate(syn["data"], spec,
                 EstConfig(smoothing=True, subarray_length=3, source_count=3))
    assert "最多分辨" in ei.value.message
    assert ei.value.details["max_resolvable_sources"] == 2


def test_rmse_decreases_with_snr_and_approaches_crb():
    """SNR 0->20dB，多次重复 RMSE 单调下降；高 SNR 段 RMSE/CRB <= 3。

    合成接口的真实噪声方差已知，故统计基准使用确定性模型的理论 CRB
    （服务对每次估计仍用特征值给出在线 CRB，见 test_crb_per_angle）。
    """
    from app.doa.crb import crb_single_source

    snrs = [0, 5, 10, 15, 20]
    k = 1000
    rmses, ratios = [], []
    for snr in snrs:
        errs = []
        spec, syn0 = make_data(angles=(18.0,), snr_db=snr, k=k, seed=1000)
        crb = crb_single_source(
            18.0, spec.positions(), 1.0, signal_power=1.0,
            noise_variance=syn0["noise_variance"], n_snapshots=k,
        )
        for seed in range(30):
            _, syn = make_data(angles=(18.0,), snr_db=snr, k=k, seed=1000 + seed)
            res = estimate(syn["data"], spec,
                           EstConfig(angle_step=0.2, source_count=1),
                           true_angles=syn["angles"])
            errs.append(res["angle_error_deg"][0])
        rmse = float(np.sqrt(np.mean(np.square(errs))))
        rmses.append(rmse)
        ratios.append(rmse / crb)
    for a, b in zip(rmses, rmses[1:]):
        assert b < a
    assert ratios[-1] <= 3.0
    # 每个估计结果里也都带在线 CRB（正数）
    res = estimate(syn0["data"], spec, EstConfig(angle_step=0.2, source_count=1))
    assert res["crb_std_deg"][0] > 0


def test_mdl_source_count_correct_high_snr():
    counts = []
    for seed in range(8):
        spec, syn = make_data(angles=(-15.0, 30.0), snr_db=20.0, k=500, seed=seed)
        res = estimate(syn["data"], spec, EstConfig(angle_step=1.0))
        counts.append(res["mdl_source_count"])
    assert counts == [2] * 8


def test_aic_and_mdl_both_reported():
    spec, syn = make_data()
    res = estimate(syn["data"], spec, EstConfig(angle_step=1.0))
    assert isinstance(res["mdl_source_count"], int)
    assert isinstance(res["aic_source_count"], int)
    assert len(res["mdl_curve"]) == 8 and len(res["aic_curve"]) == 8


def test_refinement_finer_than_step_by_order_of_magnitude():
    """细化精度至少比扫描步长细一个数量级。"""
    spec, syn = make_data(angles=(31.37,), snr_db=30.0, k=500, seed=3)
    res = estimate(syn["data"], spec,
                   EstConfig(angle_step=2.0, source_count=1),
                   true_angles=syn["angles"])
    assert res["refine_tolerance_deg"] <= 2.0 / 10.0
    assert abs(res["angle_error_deg"][0]) < 0.1


def test_crb_per_angle_present_and_positive():
    spec, syn = make_data(angles=(-25.0, 25.0))
    res = estimate(syn["data"], spec, EstConfig(angle_step=0.5),
                   true_angles=syn["angles"])
    assert all(c > 0 for c in res["crb_std_deg"])
    assert all(r is not None and r >= 0 for r in res["error_over_crb"])


def test_synthesis_bit_identical_same_seed():
    spec = ArraySpec(elements=8, spacing=0.5, wavelength=1.0)
    kw = dict(m=8, k=137, angles_deg=[-5.0, 17.0], powers=[1.0, 2.0],
              snr_db=12.0, positions=spec.positions(), wavelength=1.0, seed=2026)
    a = generate_snapshots(**kw)["data"]
    b = generate_snapshots(**kw)["data"]
    assert a.dtype == np.complex128
    assert np.array_equal(a, b)


def test_synthesis_different_seed_differs():
    spec = ArraySpec(elements=8, spacing=0.5, wavelength=1.0)
    kw = dict(m=8, k=137, angles_deg=[-5.0], powers=[1.0],
              snr_db=12.0, positions=spec.positions(), wavelength=1.0)
    a = generate_snapshots(seed=1, **kw)["data"]
    b = generate_snapshots(seed=2, **kw)["data"]
    assert not np.array_equal(a, b)


# ---- 人读错误 -----------------------------------------------------------
def test_error_source_count_ge_elements():
    spec, syn = make_data(k=64)
    with pytest.raises(SourceCountError) as ei:
        estimate(syn["data"], spec, EstConfig(source_count=8))
    assert "不小于" in ei.value.message


def test_error_fewer_snapshots_than_elements():
    spec = ArraySpec(8, 0.5, 1.0)
    x = np.zeros((8, 5), dtype=complex)
    with pytest.raises(SourceCountError) as ei:
        estimate(x, spec, EstConfig(source_count=1))
    assert "少于阵元数" in ei.value.message


def test_error_nonpositive_wavelength_or_spacing():
    with pytest.raises(ValidationError):
        ArraySpec(8, spacing=0.5, wavelength=0).positions()
    with pytest.raises(ValidationError):
        ArraySpec(8, spacing=-0.1, wavelength=1.0).positions()


def test_error_nan_in_data():
    from app.doa.engine import validate_array_inputs
    real = np.zeros((8, 20))
    imag = np.zeros((8, 20))
    imag[0, 3] = np.nan
    with pytest.raises(ValidationError, match="NaN"):
        validate_array_inputs(real, imag)


def test_error_shape_mismatch_real_imag():
    from app.doa.engine import validate_array_inputs
    with pytest.raises(ValidationError, match="形状不一致"):
        validate_array_inputs(np.zeros((8, 20)), np.zeros((8, 19)))

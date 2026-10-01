"""测向内核总装：校验 + 平滑 + 源数判定 + MUSIC + CRB + 栅瓣。

对上层只暴露 :func:`estimate` 与 :func:`validate_array_inputs`，
内核不接触会话、存储、HTTP。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import (
    LimitExceededError,
    SourceCountError,
    ValidationError,
)
from .array import element_positions, grating_alias_pairs, has_grating_ambiguity
from .covariance import hermitianize, sample_covariance
from .crb import crb_single_source
from .criteria import estimate_source_counts
from .music import music_estimate
from .smoothing import default_subarray_length, spatial_smoothing

MAX_ELEMENTS = 64
MAX_SNAPSHOTS_PER_BATCH = 10_000
MAX_SCAN_POINTS = 200_000


@dataclass(frozen=True)
class ArraySpec:
    elements: int
    spacing: float
    wavelength: float
    position_errors: tuple[float, ...] | None = None

    def positions(self) -> np.ndarray:
        return element_positions(
            self.elements,
            self.spacing,
            self.wavelength,
            np.asarray(self.position_errors, dtype=float)
            if self.position_errors is not None
            else None,
        )


@dataclass(frozen=True)
class EstConfig:
    angle_min: float = -90.0
    angle_max: float = 90.0
    angle_step: float = 1.0
    smoothing: bool = False
    subarray_length: int | None = None
    source_count: int | None = None  # None = 按 MDL（同时返回 AIC）


def validate_array_inputs(data_real: np.ndarray, data_imag: np.ndarray) -> np.ndarray:
    """实部/虚部形状与数值校验，拼回复数据 ``(M, K)``。"""
    if data_real.shape != data_imag.shape:
        raise ValidationError(
            "实部与虚部形状不一致："
            f"real={data_real.shape}, imag={data_imag.shape}",
            details={"real_shape": list(data_real.shape), "imag_shape": list(data_imag.shape)},
        )
    if data_real.ndim != 2:
        raise ValidationError(f"数据必须是二维 (M, K)，收到 {data_real.ndim} 维")
    if not (np.all(np.isfinite(data_real)) and np.all(np.isfinite(data_imag))):
        raise ValidationError("数据中存在 NaN 或 Inf")
    return data_real + 1j * data_imag


def _check_limits(spec: ArraySpec, cfg: EstConfig, k: int,
                  check_snapshot_limit: bool = True) -> None:
    if spec.elements > MAX_ELEMENTS:
        raise LimitExceededError(
            f"阵元数 {spec.elements} 超过上限 {MAX_ELEMENTS}",
            details={"value": spec.elements, "max": MAX_ELEMENTS},
        )
    # 单批快拍上限只针对“本次提交的一批”；会话累计协方差路径
    # （尤其指数遗忘）累计条数允许超过该值，故此处可关闭。
    if check_snapshot_limit and k > MAX_SNAPSHOTS_PER_BATCH:
        raise LimitExceededError(
            f"单批快拍数 {k} 超过上限 {MAX_SNAPSHOTS_PER_BATCH}",
            details={"value": k, "max": MAX_SNAPSHOTS_PER_BATCH},
        )
    n_grid = int((cfg.angle_max - cfg.angle_min) / cfg.angle_step) + 1
    if n_grid > MAX_SCAN_POINTS:
        raise LimitExceededError(
            f"扫描点数 {n_grid} 超过上限 {MAX_SCAN_POINTS}"
            f"（范围 {cfg.angle_min}..{cfg.angle_max}，步长 {cfg.angle_step}）",
            details={"value": n_grid, "max": MAX_SCAN_POINTS},
        )
    if cfg.angle_min < -90.0 or cfg.angle_max > 90.0 or cfg.angle_max <= cfg.angle_min:
        raise ValidationError("角度范围必须满足 -90 <= min < max <= 90（度）")
    if cfg.angle_step <= 0:
        raise ValidationError("扫描步长必须为正")


def estimate(
    data: np.ndarray | None,
    spec: ArraySpec,
    cfg: EstConfig,
    *,
    covariance: np.ndarray | None = None,
    n_snapshots: int | None = None,
    effective_snapshots: int | None = None,
    true_angles: list[float] | None = None,
) -> dict:
    """运行一次完整估计。

    两种输入方式（二选一）：
      * 直接给 ``data``：``(M, K)`` 复快拍，内部算样本协方差；
      * 会话路径：给 ``covariance`` 与 ``n_snapshots``（增量累加结果）。
    """
    positions = spec.positions()
    m = spec.elements
    is_covariance_path = data is None

    if data is not None:
        if data.shape[0] != m:
            raise ValidationError(
                f"数据阵元维 {data.shape[0]} 与配置 M={m} 不一致"
            )
        k = data.shape[1]
        _check_limits(spec, cfg, k, check_snapshot_limit=True)
        if k < m:
            # 与“第一次出估计时累计快拍还没阵元多”对应
            raise SourceCountError(
                f"累计快拍数 {k} 少于阵元数 {m}，无法估计",
                details={"snapshots": k, "elements": m},
            )
        r_full = sample_covariance(data)
    else:
        if covariance is None or n_snapshots is None:
            raise ValidationError("必须提供 data 或 covariance+n_snapshots")
        if covariance.shape != (m, m):
            raise ValidationError(
                f"协方差形状 {covariance.shape} 与 M={m} 不匹配"
            )
        k = int(n_snapshots)
        if k < m:
            raise SourceCountError(
                f"累计快拍数 {k} 少于阵元数 {m}，无法估计",
                details={"snapshots": k, "elements": m},
            )
        r_full = hermitianize(np.asarray(covariance, dtype=np.complex128))
    # 阵列/扫描范围等通用上限（累计快拍的单批上限不在这里约束）
    _check_limits(spec, cfg, k, check_snapshot_limit=False)

    # ---- 空间平滑 ----
    l = cfg.subarray_length
    if cfg.smoothing:
        if spec.position_errors is not None:
            raise ValidationError(
                "存在阵元位置误差时平移不变性被破坏，不能使用空间平滑"
            )
        if l is None:
            l = default_subarray_length(m)
        if l < 2 or l > m:
            raise ValidationError(
                f"子阵长度 L 必须满足 2 <= L <= M={m}，收到 L={l}"
            )
        j = m - l + 1
        d_max = min(l - 1, 2 * j)
        # 显式指定信源数时，提前给出可读懂的上限错误
        if cfg.source_count is not None and cfg.source_count > d_max:
            raise SourceCountError(
                f"子阵长度 L={l}（前后向各 {j} 个子阵）最多分辨 "
                f"{d_max} 个信源，无法估计 {int(cfg.source_count)} 个；"
                "请增大 L 或减少信源数",
                details={
                    "subarray_length": l,
                    "n_subarrays_each": j,
                    "max_resolvable_sources": d_max,
                    "requested_sources": int(cfg.source_count),
                },
            )
        r_work, n_sub = spatial_smoothing(r_full, l)
        # 平滑的等效快拍数 = 每快拍子阵数 × 统计有效快拍数
        n_eff = n_sub * int(effective_snapshots or k)
        # 子阵仍为同间距 ULA，整体平移不影响 MUSIC 与 CRB 孔径，直接以
        # 自身中心为原点构造位置即可。
        positions_work = (np.arange(l, dtype=float) - (l - 1) / 2.0) * spec.spacing
        m_work = l
    else:
        d_max = m - 1
        r_work = r_full
        n_eff = int(effective_snapshots or k)
        positions_work = positions
        m_work = m
        l = None

    # ---- 源数判定 ----
    crit = estimate_source_counts(r_work, n_eff)
    if cfg.source_count is not None:
        d = int(cfg.source_count)
        if d <= 0:
            raise SourceCountError(f"信源数必须为正整数，收到 {d}")
        if d >= m_work:
            raise SourceCountError(
                f"信源数 {d} 不小于有效阵元数 {m_work}（M{'子阵' if cfg.smoothing else ''}={m_work}）",
                details={"sources": d, "effective_elements": m_work},
            )
        if d > d_max:
            # 平滑路径的上限已在 validate_smoothing 抛；此处兜底
            raise SourceCountError(
                f"信源数 {d} 超过当前配置可分辨上限 {d_max}",
                details={"sources": d, "max_resolvable": d_max},
            )
        source_count_source = "specified"
    else:
        d = int(crit["mdl"])
        if d >= m_work:
            raise SourceCountError(
                f"MDL 判定信源数为 {d}，不小于有效阵元数 {m_work}；"
                "请显式指定 source_count 或改善数据质量"
            )
        if d == 0:
            d = max(crit["aic"], 1) if crit["aic"] < m_work else 1
            source_count_source = "mdl_zero_fallback_aic"
        else:
            source_count_source = "mdl"
        if cfg.smoothing and d > d_max:
            raise SourceCountError(
                f"MDL 判定信源数为 {d}，超过子阵长度 L={l} 的可分辨上限 "
                f"{d_max}；请增大 L、减少信源数或显式指定 source_count",
                details={
                    "sources": d,
                    "subarray_length": l,
                    "max_resolvable_sources": d_max,
                },
            )

    # ---- MUSIC ----
    music = music_estimate(
        r_work, d, positions_work, spec.wavelength,
        angle_min=cfg.angle_min, angle_max=cfg.angle_max,
        angle_step=cfg.angle_step,
    )

    # ---- 功率/噪声估计：由最终协方差特征值给出 ----
    # 样本协方差的噪声特征值对 sigma2 是一致估计但小样本有偏，按
    # 有效快拍/自由度做一阶修正，避免在线 CRB 系统性偏小。
    evals_desc = np.linalg.eigvalsh(hermitianize(r_work))[::-1]
    noise_floor = float(np.mean(evals_desc[d:])) if d < m_work else float(evals_desc[-1])
    noise_est = noise_floor * n_eff / max(n_eff - m_work - d, 1)
    # 确定性模型下信号特征值约 P + sigma2
    signal_powers = [
        float(max(evals_desc[i] - noise_est, np.finfo(float).eps)) for i in range(d)
    ]

    # ---- 每角 CRB（按单源白噪声确定性模型）与比值 ----
    crbs, ratios, errors = [], [], []
    for i, ang in enumerate(music["angles"]):
        crb_sd = crb_single_source(
            ang, positions_work, spec.wavelength,
            signal_power=signal_powers[i],
            noise_variance=max(noise_est, np.finfo(float).eps),
            n_snapshots=n_eff,
        )
        crbs.append(crb_sd)
        if true_angles is not None and i < len(true_angles):
            err = float(ang) - float(true_angles[i])
            errors.append(err)
            ratios.append(abs(err) / crb_sd if crb_sd > 0 else None)
        else:
            ratios.append(None)

    # ---- 栅瓣模糊 ----
    grating = has_grating_ambiguity(spec.spacing, spec.wavelength)
    alias_pairs = (
        grating_alias_pairs(
            positions, spec.wavelength,
            cfg.angle_min, cfg.angle_max,
            angle_step=max(cfg.angle_step, 0.5),
        )
        if grating
        else []
    )

    return {
        "angles": music["angles"],
        "source_count": d,
        "source_count_source": source_count_source,
        "mdl_source_count": crit["mdl"],
        "aic_source_count": crit["aic"],
        "mdl_curve": crit["mdl_curve"],
        "aic_curve": crit["aic_curve"],
        "eigenvalues": crit["eigenvalues"],
        "resolved": music["resolved"],
        "n_peaks_found": music["n_peaks_found"],
        "spectrum": music["spectrum"],
        "spectrum_angles_grid": music["grid"],
        "spectrum_at_angles": music["spectrum_at_angles"],
        "refine_tolerance_deg": music["refine_tolerance_deg"],
        "refine_brackets": music["refine_brackets"],
        "crb_std_deg": crbs,
        "error_over_crb": ratios,
        "angle_error_deg": errors if true_angles is not None else None,
        "signal_power_est": signal_powers,
        "noise_variance_est": noise_est,
        "n_snapshots": int(k),
        "n_effective_snapshots": int(n_eff),
        "smoothing": {
            "enabled": cfg.smoothing,
            "subarray_length": l,
            "max_resolvable_sources": d_max,
        },
        "grating_ambiguity": {
            "present": bool(grating),
            "spacing_over_half_wavelength": spec.spacing / spec.wavelength,
            "alias_pairs": [list(p) for p in alias_pairs],
        },
    }

"""单信源、白噪声、确定性（条件）模型下的克拉美-罗界。

对 ULA（含阵元位置误差，只要沿轴向）单源情形，DOA 的 CRB 方差为

    var(theta) = (1/2) * sigma2 / (N * P) / sum_m (w_m^2 - wbar^2)

其中 ``w_m = 2 pi p_m cos(theta)/lambda``，``wbar`` 为其算术平均。
单位为弧度平方，接口返回度。

参考：Stoica & Nehorai, "MUSIC, maximum likelihood, and Cramer-Rao
bound", IEEE TASSP 37(5), 1989（单源退化形式）。
"""

from __future__ import annotations

import math

import numpy as np


def crb_single_source(
    angle_deg: float,
    positions: np.ndarray,
    wavelength: float,
    signal_power: float,
    noise_variance: float,
    n_snapshots: int,
) -> float:
    """返回单源 DOA 估计的 CRB 标准差（度）。"""
    if signal_power <= 0:
        raise ValueError("信号功率必须为正")
    if noise_variance <= 0:
        raise ValueError("噪声方差必须为正")
    if n_snapshots <= 0:
        raise ValueError("快拍数必须为正")
    cos_t = math.cos(math.radians(angle_deg))
    w = (2.0 * math.pi / wavelength) * positions * cos_t
    wbar = float(np.mean(w))
    spatial_var = float(np.sum((w - wbar) ** 2))
    if spatial_var <= 0:
        raise ValueError("阵列孔径为零，无法计算 CRB")
    var_rad2 = 0.5 * (noise_variance / (n_snapshots * signal_power)) / spatial_var
    return math.degrees(math.sqrt(var_rad2))


def crb_variance_deg2(*args, **kwargs) -> float:
    """返回 CRB 方差（度平方）。"""
    return crb_single_source(**kwargs) ** 2

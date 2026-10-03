"""常规波束形成（Bartlett / delay-and-sum）功率估计。

航迹关联需要“每个估计角度对应的相对功率”作为辅助线索。内核引擎给出
的 ``signal_power_est`` 按特征值降序排列，与角度无对应关系，不能直接
用。这里给定向量归一化后计算

    P(theta) = a(theta)^H R a(theta) / M

它是入射方向上的波束输出功率（含该方向上其他源经方向图泄漏的部分），
对分离良好的源与真实功率单调对应；交叉时两方向图几乎重合，该量本就
不具备分辨力（由航迹层的交叉保持逻辑处理，而不是强行按功率对号）。
"""

from __future__ import annotations

import numpy as np

from .array import steering_vector


def beamscan_powers(
    angles_deg,
    covariance: np.ndarray,
    positions: np.ndarray,
    wavelength: float,
) -> list[float]:
    """逐角度的归一化波束功率。无角度时返回空表。"""
    angles = np.asarray(angles_deg, dtype=float)
    if angles.size == 0:
        return []
    r = (covariance + covariance.conj().T) * 0.5
    a = steering_vector(angles, positions, wavelength)  # (M, G)
    m = a.shape[0]
    # 归一化方向向量，消除阵列增益对角度的轻微依赖
    norm = np.sqrt(np.einsum("gi,gi->i", a.conj(), a).real)
    an = a / norm[None, :]
    powers = np.einsum("gi,gh,hi->i", an.conj(), r, an).real / m
    return [float(p) for p in powers]

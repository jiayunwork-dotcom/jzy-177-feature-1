"""闭环自调用合成数据生成。

模型（窄带、远场、确定性方向向量 + 白噪声）：

    X[:, k] = A(theta) @ s[:, k] + n[:, k]
    n[:, k] ~ CN(0, sigma2 I),  sigma2 = P_ref * 10^(-SNR/10)

``P_ref`` 取非相干源功率之和（相干源的信号幅度按各源功率直接给定，
同一波形线性叠加，相干组合的功率随角度改变——这是相干源的物理
本来面目，与空间平滑的处理对象一致）。

随机数使用 ``np.random.Generator(PCG64(seed))``：同一种子、同一
参数下，严格按固定顺序抽样，逐位一致。
"""

from __future__ import annotations

import math

import numpy as np

from .array import steering_vector


def _check2d_finite(x: np.ndarray, name: str, m: int) -> np.ndarray:
    if x.shape[0] != m:
        raise ValueError(f"{name} 第一维必须等于阵元数 M={m}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} 中存在 NaN 或 Inf")
    return x


def generate_snapshots(
    m: int,
    k: int,
    angles_deg: list[float],
    powers: list[float],
    snr_db: float,
    positions: np.ndarray,
    wavelength: float,
    seed: int,
    coherent: bool = False,
) -> dict:
    """生成长度 K 的快拍。

    返回 ``{"data": (M,K) complex128, "noise_variance": float,
    "powers": [...], "angles": [...]}``。
    ``coherent=True`` 时所有源共用同一个 CN(0,1) 波形；否则各源独立。
    抽样顺序固定：先信号（逐源抽 K 个复高斯），再噪声（M*K 个复高斯），
    实部、虚部各半方差，保证 ``CN(0, v)`` 约定。
    """
    if m <= 0 or k <= 0:
        raise ValueError("阵元数与快拍数必须为正")
    if len(angles_deg) != len(powers):
        raise ValueError("angles 与 powers 长度必须一致")
    if any(p <= 0 for p in powers):
        raise ValueError("信源功率必须为正")

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    d = len(angles_deg)

    def cn(shape):
        # CN(0,1)：实虚部独立 N(0, 1/2)
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2.0)

    if coherent and d > 0:
        waveform = cn((1, k))
        s = np.vstack([math.sqrt(p) * waveform for p in powers])
    else:
        s = np.vstack([math.sqrt(p) * cn(k) for p in powers])

    a = steering_vector(np.asarray(angles_deg, dtype=float), positions, wavelength)
    ref_power = float(sum(powers))
    noise_var = ref_power * 10.0 ** (-snr_db / 10.0)
    n = math.sqrt(noise_var) * cn((m, k))

    data = a @ s + n
    return {
        "data": data.astype(np.complex128),
        "noise_variance": noise_var,
        "powers": [float(p) for p in powers],
        "angles": [float(x) for x in angles_deg],
        "coherent": bool(coherent),
    }

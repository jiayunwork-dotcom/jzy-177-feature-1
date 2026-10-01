"""协方差矩阵：批量样本协方差、厄米化与数值工具。"""

from __future__ import annotations

import numpy as np


def sample_covariance(data: np.ndarray) -> np.ndarray:
    """由 ``(M, K)`` 复快拍计算样本协方差 ``R = X X^H / K``（无偏归一）。

    这里除以快拍数 K（样本协方差），与增量累加器的定义保持一致。
    """
    if data.ndim != 2:
        raise ValueError(f"数据必须是二维 (M, K)，收到形状 {data.shape}")
    k = data.shape[1]
    r = (data @ data.conj().T) * (1.0 / k)
    return hermitianize(r)


def hermitianize(r: np.ndarray) -> np.ndarray:
    """强制厄米对称性，消除逐批累加引入的不对称舍入尾差。"""
    return (r + r.conj().T) * 0.5

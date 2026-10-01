"""信源数判定：MDL / AIC（Wax & Kailath 形式）。

由样本协方差的特征值（升序）与快拍数计算两种准则，取使准则最小的
信源个数。MDL 大样本一致，AIC 快拍少时更激进（倾向高估）。
"""

from __future__ import annotations

import numpy as np


def _aic_mdl(eigenvalues_asc: np.ndarray, n_snapshots: int) -> tuple[np.ndarray, np.ndarray]:
    """计算 k = 0 .. M-1 的 AIC、MDL 值（升序特征值，噪声在最前）。"""
    ev = np.asarray(eigenvalues_asc, dtype=np.float64)
    m = ev.size
    n = float(n_snapshots)
    aic = np.empty(m, dtype=np.float64)
    mdl = np.empty(m, dtype=np.float64)
    for k in range(m):
        noise = ev[: m - k]  # 假设 k 个信号，其余 m-k 个为噪声特征值
        q = noise.size
        if q == 0:
            # k == m-1 时仅剩 1 个噪声特征值，几何/算术平均恒等，
            # log 比值取 0；参数惩罚仍正常计入。
            log_ratio = 0.0
        else:
            geo = np.exp(np.mean(np.log(np.maximum(noise, np.finfo(float).tiny))))
            arith = float(np.mean(noise))
            # log(geo/arith) <= 0；似然项 -2N * q * log(geo/arith) >= 0
            log_ratio = q * (
                np.log(max(geo, np.finfo(float).tiny))
                - np.log(max(arith, np.finfo(float).tiny))
            )
        params = k * (2 * m - k) + 1
        aic[k] = -2.0 * n * log_ratio + 2.0 * params
        mdl[k] = -n * log_ratio + 0.5 * params * np.log(max(n, 1.0))
    return aic, mdl


def estimate_source_counts(
    r: np.ndarray, n_snapshots: int
) -> dict:
    """返回 MDL 与 AIC 各自给出的信源数及完整准则曲线。

    返回字段：``mdl``、``aic``（信源数），``mdl_curve``、``aic_curve``
    （按 k=0..M-1 排列的准则值），``eigenvalues``（升序特征值）。
    """
    evals = np.linalg.eigvalsh((r + r.conj().T) * 0.5)[::-1]  # 降序
    asc = evals[::-1]
    aic, mdl = _aic_mdl(asc, n_snapshots)
    return {
        "mdl": int(np.argmin(mdl)),
        "aic": int(np.argmin(aic)),
        "mdl_curve": mdl.tolist(),
        "aic_curve": aic.tolist(),
        "eigenvalues": evals.tolist(),
    }

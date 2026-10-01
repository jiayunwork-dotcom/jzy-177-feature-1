"""前后向空间平滑（Forward/Backward Spatial Smoothing）。

只需要完整阵列的样本协方差 R 即可（切子阵 = 取主子阵），因此与
流式累加器天然兼容，不必逐快拍缓存。
"""

from __future__ import annotations

import numpy as np

from ..errors import SourceCountError, ValidationError
from .covariance import hermitianize

_J = np.array([[0, 1], [1, 0]], dtype=np.float64)


def exchange_matrix(m: int) -> np.ndarray:
    """m 阶交换矩阵 J（反对角线为 1）。"""
    return np.eye(m, dtype=np.float64)[:, ::-1]


def default_subarray_length(m: int) -> int:
    """默认子阵长度 L：取使 2(M-L+1) >= L-1 成立的最大 L。

    该选择让最大可分辨信源数 ``min(L-1, 2J)`` 最大（约 ``2M/3``）。
    """
    # 2(M-L+1) >= L-1  <=>  3L <= 2M+3
    l = int((2 * m + 3) // 3)
    return max(2, min(l, m))


def validate_smoothing(m: int, subarray_length: int, n_sources: int) -> int:
    """校验平滑配置，返回该配置下最多可分辨的信源数。

    超限时抛出 :class:`SourceCountError`，错误信息中带上限值。
    """
    if subarray_length < 2 or subarray_length > m:
        raise ValidationError(
            f"子阵长度 L 必须满足 2 <= L <= M={m}，收到 L={subarray_length}"
        )
    j = m - subarray_length + 1
    # 子阵阵元数约束 L > d；前后向子阵数量约束 d <= 2J
    d_max = min(subarray_length - 1, 2 * j)
    if n_sources > d_max:
        raise SourceCountError(
            f"子阵长度 L={subarray_length}（前后向各 {j} 个子阵）最多分辨 "
            f"{d_max} 个信源，无法估计 {n_sources} 个；请增大 L 或减少信源数",
            details={
                "subarray_length": subarray_length,
                "n_subarrays_each": j,
                "max_resolvable_sources": d_max,
                "requested_sources": n_sources,
            },
        )
    return d_max


def spatial_smoothing(r: np.ndarray, subarray_length: int) -> tuple[np.ndarray, int]:
    """对完整阵列协方差做前后向空间平滑。

    返回 ``(R_fb, n_eff)``，其中 ``n_eff = 2 J K`` 是平滑后等效快拍数
    （K 为形成 R 时的快拍数，由调用方另算；这里只返回每个快拍的子阵
    数量 ``2J`` 的系数，实际等效数 = ``2J * K``）。

    前向：``R_f = (1/J) sum_{p=0}^{J-1} R[p:p+L, p:p+L]``
    后向：``R_b = J R_f^* J``
    """
    m = r.shape[0]
    l = subarray_length
    if l > m:
        raise ValidationError(f"子阵长度 L={l} 大于阵元数 M={m}")
    j = m - l + 1
    r_f = np.zeros((l, l), dtype=np.complex128)
    for p in range(j):
        r_f += r[p : p + l, p : p + l]
    r_f /= j
    jmat = exchange_matrix(l)
    r_b = jmat @ r_f.conj() @ jmat
    r_fb = hermitianize((r_f + r_b) * 0.5)
    return r_fb, 2 * j

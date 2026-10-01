"""流式协方差累加器：定长滑动窗口 与 指数遗忘因子。

两种定义都与“对同一份快拍按同一加权定义一次算出”的结果保持一致：

* 滑动窗口（``mode="sliding"``）：
      R = (1/W) sum_{k in window} x_k x_k^H
  增量维护未归一化和 S 与环形容器；窗口满后每进一批、被挤出窗口的
  旧快拍直接扣除。为避免无限追加带来浮点漂移，每当有快拍被挤出时
  都由窗口内现存快拍重建一次 S（W <= 10000，开销可接受）。

* 指数遗忘（``mode="forgetting"``）：
      对“最近 N 条快拍”的截断几何加权样本均值（w_i = lambda^(N-i)，
      i=1..N）。用一个固定的维护规则：每追加一条，
          S' = lambda S + x x^H,  w' = lambda w + 1,  R = S / w
  截断几何权重由数学归纳法自动满足（最老快拍权重为 lambda^(N-1)），
  无需历史快拍。w 被夹在 1e-300 以下也不会下溢（N 有限、lambda<=1）。

每次更新后对 R/S 做厄米对称化，保证长期运行不丢厄米性。
"""

from __future__ import annotations

import numpy as np

from ..errors import ValidationError


class CovarianceAccumulator:
    def __init__(
        self,
        m: int,
        mode: str = "sliding",
        window_size: int | None = 500,
        forgetting_factor: float | None = None,
    ):
        if mode not in ("sliding", "forgetting"):
            raise ValidationError(f"未知协方差更新方式：{mode!r}（支持 sliding/forgetting）")
        self.m = int(m)
        self.mode = mode
        if mode == "sliding":
            if not window_size or int(window_size) <= 0:
                raise ValidationError("sliding 模式要求 window_size 为正整数")
            self.window_size = int(window_size)
            self.forgetting_factor = None
        else:
            ff = 0.98 if forgetting_factor is None else float(forgetting_factor)
            if not (0.0 < ff <= 1.0):
                raise ValidationError("遗忘因子 forgetting_factor 必须满足 0 < lambda <= 1")
            self.forgetting_factor = ff
            self.window_size = None
        # 未归一化加权和
        self._s = np.zeros((m, m), dtype=np.complex128)
        # 权重和（forgetting）；sliding 即窗口内快拍数
        self._w = 0.0
        # sliding 的环形缓冲
        self._buffer: np.ndarray | None = (
            np.zeros((self.window_size, m), dtype=np.complex128)  # type: ignore[arg-type]
            if mode == "sliding"
            else None
        )
        self._head = 0      # 下一个写入位置
        self._filled = 0    # 缓冲中已有快拍数
        self.total_appended = 0  # 历史累计接收条数（去重后的）

    # ------------------------------------------------------------------
    @property
    def n_snapshots(self) -> int:
        """当前 R 定义中实际计入的快拍数（窗口内 / 已接收总数）。"""
        if self.mode == "sliding":
            return self._filled
        return self.total_appended

    @property
    def effective_snapshots(self) -> int:
        """用于源数判定/CRB 的等效独立快拍数。

        滑动窗口即窗口内条数；指数遗忘下几何加权的“有效样本量”约为
        权重和的平方除以权重平方和（Kish 形式）：

            N_eff = (Σw)^2 / Σw^2 = (1−λ^N)^2 / ((1−λ^2N)(1−λ)/(1+λ))

        稳态 N→∞ 时收敛到 (1+λ)/(1−λ)。
        """
        if self.mode == "sliding":
            return self._filled
        lam = self.forgetting_factor
        n = self.total_appended
        if lam is None or lam >= 1.0 or n <= 0:
            return n
        a = 1.0 - lam ** n
        b = 1.0 - lam ** (2 * n)
        if b <= 0:
            return n
        n_eff = a * a * (1.0 + lam) / (b * (1.0 - lam))
        return max(1, int(round(n_eff)))

    def covariance(self) -> np.ndarray:
        if self._w <= 0:
            return (self._s + self._s.conj().T) * 0.5
        r = self._s / self._w
        return (r + r.conj().T) * 0.5

    # ------------------------------------------------------------------
    def add_batch(self, data: np.ndarray) -> None:
        """追加一批 ``(M, K)`` 复快拍（K 可逐批不同，不超过窗口上限）。"""
        if data.ndim != 2 or data.shape[0] != self.m:
            raise ValidationError(
                f"快拍形状必须为 (M={self.m}, K)，收到 {data.shape}"
            )
        if not np.all(np.isfinite(data)):
            raise ValidationError("快拍数据中存在 NaN 或 Inf")
        if self.mode == "sliding":
            self._add_sliding(np.ascontiguousarray(data.T))  # (K, M)
        else:
            self._add_forgetting(np.ascontiguousarray(data.T))
        self.total_appended += data.shape[1]
        # 厄米兜底：对未归一化和做对称化（归一化的对称性随之保持）
        self._s = (self._s + self._s.conj().T) * 0.5

    def _add_sliding(self, rows: np.ndarray) -> None:
        """rows: (K, M)。逐批写入环形缓冲；只要本批触发过驱逐，
        最后就由窗口内现存快拍重建一次 S（同时杜绝扣减法的漂移）。"""
        assert self._buffer is not None
        k = rows.shape[0]
        evicted = self._filled + k > self.window_size
        for x in rows:
            self._buffer[self._head] = x
            self._head = (self._head + 1) % self.window_size
            self._filled = min(self._filled + 1, self.window_size)
        if evicted:
            # 由窗口内现存快拍重建未归一化和；集合即正确窗口内容，
            # 与排列顺序无关，重复重建无数值漂移。
            # 行存 x_k (W,M)：sum x_k x_k^H = B^T @ conj(B)。
            active = self._buffer[: self._filled]
            self._s = active.T @ active.conj()
        else:
            self._s += rows.T @ rows.conj()
        self._w = float(self._filled)

    def _add_forgetting(self, rows: np.ndarray) -> None:
        """逐条 S <- lambda S + x x^H；批量用矩阵乘法一次算新增部分。

        设本批 K 条 x_1..x_K，则
            S_new = lambda^K S_old + sum_{i=1}^K lambda^(K-i) x_i x_i^H
        按定义逐条更新等价；为与“逐条顺序追加”严格一致（含同一舍入
        次序），仍逐条循环，但每条只做一次秩-1 更新。
        """
        lam = self.forgetting_factor
        assert lam is not None
        w = self._w
        for x in rows:
            self._s *= lam
            self._s += np.outer(x, x.conj())
            w = lam * w + 1.0
        self._w = w

    # ------------------------------------------------------------------
    def to_state(self) -> dict:
        """小体积元数据（缓冲另走 :meth:`pack_buffer` 二进制落盘）。"""
        return {
            "mode": self.mode,
            "window_size": self.window_size,
            "forgetting_factor": self.forgetting_factor,
            "s_real": np.real(self._s).tolist(),
            "s_imag": np.imag(self._s).tolist(),
            "w": self._w,
            "head": self._head,
            "filled": self._filled,
            "total_appended": self.total_appended,
        }

    def pack_buffer(self) -> dict[str, np.ndarray] | None:
        """滑动窗口环形缓冲的实/虚部数组（供 np.savez 二进制落盘）。"""
        if self._buffer is None:
            return None
        return {"real": np.real(self._buffer), "imag": np.imag(self._buffer)}

    @classmethod
    def from_state(
        cls,
        state: dict,
        buffer_real: np.ndarray | None = None,
        buffer_imag: np.ndarray | None = None,
    ) -> "CovarianceAccumulator":
        acc = cls(
            m=int(np.asarray(state["s_real"]).shape[0]),
            mode=state["mode"],
            window_size=state.get("window_size"),
            forgetting_factor=state.get("forgetting_factor"),
        )
        acc._s = (
            np.asarray(state["s_real"], dtype=np.float64)
            + 1j * np.asarray(state["s_imag"], dtype=np.float64)
        )
        acc._w = float(state["w"])
        acc._head = int(state["head"])
        acc._filled = int(state["filled"])
        acc.total_appended = int(state["total_appended"])
        if buffer_real is not None and buffer_imag is not None:
            acc._buffer = (
                np.asarray(buffer_real, dtype=np.float64)
                + 1j * np.asarray(buffer_imag, dtype=np.float64)
            )
        return acc


# ----------------------------------------------------------------------
# 批处理参考实现（供测试与“切批追加 == 一次送入”等价性核对）
# ----------------------------------------------------------------------
def batch_sliding_covariance(data: np.ndarray, window_size: int) -> tuple[np.ndarray, int]:
    """与累加器相同定义的滑动窗口批处理参考。"""
    window = data[:, -window_size:]
    k = window.shape[1]
    r = (window @ window.conj().T) / k
    r = (r + r.conj().T) * 0.5
    return r, k


def batch_forgetting_covariance(
    data: np.ndarray, forgetting_factor: float
) -> tuple[np.ndarray, int]:
    """截断几何加权样本均值的批处理参考（与逐条递推同一定义）。"""
    k = data.shape[1]
    lam = forgetting_factor
    weights = lam ** (k - 1 - np.arange(k))  # 最老 -> 最新
    # 等价于 wsum = 1 + lam + ... + lam^(K-1)
    xw = data * np.sqrt(weights)[None, :]
    s = xw @ xw.conj().T
    wsum = float(np.sum(weights))
    r = s / wsum
    r = (r + r.conj().T) * 0.5
    return r, k

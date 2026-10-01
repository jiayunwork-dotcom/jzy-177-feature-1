"""MUSIC 空间谱：粗扫、找峰、峰附近细化。"""

from __future__ import annotations

import numpy as np

from .array import steering_vector


def noise_subspace(r: np.ndarray, n_sources: int) -> np.ndarray:
    """取协方差最小的 M-d 个特征向量作为噪声子空间 ``(M, M-d)``。"""
    _, evecs = np.linalg.eigh((r + r.conj().T) * 0.5)
    return evecs[:, : r.shape[0] - n_sources]


def music_spectrum(angles_deg: np.ndarray, positions: np.ndarray,
                   wavelength: float, en: np.ndarray) -> np.ndarray:
    """在给定角度栅格上计算 MUSIC 谱 ``1 / (a^H En En^H a)``。"""
    a = steering_vector(angles_deg, positions, wavelength)  # (M, G)
    proj = en @ en.conj().T
    denom = np.einsum("gi,gh,hi->i", a.conj(), proj, a).real
    denom = np.maximum(denom, np.finfo(np.float64).tiny)
    return 1.0 / denom


def _golden_refine(theta_lo: float, theta_hi: float, positions: np.ndarray,
                   wavelength: float, en: np.ndarray, tol: float,
                   max_iter: int = 120) -> float:
    """在 ``[lo, hi]`` 内用黄金分割最大化谱（即最小化分母）。"""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0  # 0.618
    a = positions
    wl = wavelength
    proj = en @ en.conj().T

    def cost(theta: float) -> float:
        sv = steering_vector(theta, a, wl)
        c = np.real(sv.conj() @ proj @ sv)
        return max(float(c), np.finfo(np.float64).tiny)

    x1 = theta_hi - inv_phi * (theta_hi - theta_lo)
    x2 = theta_lo + inv_phi * (theta_hi - theta_lo)
    f1, f2 = cost(x1), cost(x2)
    for _ in range(max_iter):
        if abs(theta_hi - theta_lo) < tol:
            break
        if f2 < f1:  # 谱更大（分母更小）
            theta_lo = x1
            x1, f1 = x2, f2
            x2 = theta_lo + inv_phi * (theta_hi - theta_lo)
            f2 = cost(x2)
        else:
            theta_hi = x2
            x2, f2 = x1, f1
            x1 = theta_hi - inv_phi * (theta_hi - theta_lo)
            f1 = cost(x1)
    mid = 0.5 * (theta_lo + theta_hi)
    # 与两端比较，返回区间内谱最高点
    candidates = [(cost(mid), mid), (f1, x1), (f2, x2)]
    return min(candidates, key=lambda t: t[0])[1]


def find_peaks(spectrum: np.ndarray) -> np.ndarray:
    """返回严格局部极大的栅格下标（边界点按单侧判断，视为候选峰）。"""
    g = spectrum.size
    if g < 3:
        return np.arange(g)
    peaks = []
    if spectrum[0] > spectrum[1]:
        peaks.append(0)
    for i in range(1, g - 1):
        if spectrum[i] > spectrum[i - 1] and spectrum[i] >= spectrum[i + 1]:
            peaks.append(i)
    if spectrum[-1] > spectrum[-2]:
        peaks.append(g - 1)
    return np.asarray(peaks, dtype=int)


def music_estimate(
    r: np.ndarray,
    n_sources: int,
    positions: np.ndarray,
    wavelength: float,
    angle_min: float = -90.0,
    angle_max: float = 90.0,
    angle_step: float = 1.0,
) -> dict:
    """完整 MUSIC 估计：粗扫 + 取前 d 个峰 + 黄金分割细化。

    细化精度为 ``angle_step / 100``（比步长细两个数量级，满足
    “至少细一个数量级”的要求）。返回角度（度，升序）、谱值、
    细化区间与粗扫栅格/谱（便于绘图核对）。
    """
    if angle_max <= angle_min:
        raise ValueError("angle_max 必须大于 angle_min")
    if angle_step <= 0:
        raise ValueError("angle_step 必须为正数")
    grid = np.arange(angle_min, angle_max + 0.5 * angle_step, angle_step)
    en = noise_subspace(r, n_sources)
    spectrum = music_spectrum(grid, positions, wavelength, en)

    peak_idx = find_peaks(spectrum)
    # 若谱形态异常找不到任何峰（理论上不会），退化为取最大点
    if peak_idx.size == 0:
        peak_idx = np.array([int(np.argmax(spectrum))])
    top = peak_idx[np.argsort(spectrum[peak_idx])[::-1][:n_sources]]
    top = top[np.argsort(grid[top])]

    tol = angle_step / 100.0
    refined_angles: list[float] = []
    refined_spectrum: list[float] = []
    brackets: list[tuple[float, float]] = []
    for idx in top:
        lo = max(angle_min, float(grid[idx] - angle_step))
        hi = min(angle_max, float(grid[idx] + angle_step))
        theta = _golden_refine(lo, hi, positions, wavelength, en, tol)
        refined_angles.append(float(theta))
        brackets.append((lo, hi))
    # 细化后按角度重排
    order = np.argsort(refined_angles)
    refined_angles = [refined_angles[i] for i in order]
    brackets = [brackets[i] for i in order]
    # 谱值在细化角度上重算
    spec_refined = music_spectrum(
        np.asarray(refined_angles), positions, wavelength, en
    ).tolist()
    return {
        "angles": refined_angles,
        "n_peaks_found": int(peak_idx.size),
        "resolved": int(len(refined_angles)) == n_sources and int(peak_idx.size) >= n_sources,
        "spectrum_at_angles": [spec_refined[i] for i in order],
        "refine_brackets": brackets,
        "grid": grid.tolist(),
        "spectrum": spectrum.tolist(),
        "refine_tolerance_deg": float(tol),
    }

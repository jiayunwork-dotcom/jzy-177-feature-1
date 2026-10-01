"""阵列几何：阵元位置、方向向量与栅瓣模糊判定。"""

from __future__ import annotations

import math

import numpy as np

from ..errors import ValidationError


def element_positions(
    m: int, spacing: float, wavelength: float, position_errors: np.ndarray | None = None
) -> np.ndarray:
    """返回以阵列中心为原点的阵元轴向位置（单位：米），长度 ``m``。

    理想线阵位置为 ``((0..M-1) - (M-1)/2) * spacing``；``position_errors``
    为沿阵列轴向的附加偏差（米）。
    """
    if wavelength <= 0:
        raise ValidationError(f"载波波长必须为正数，收到 {wavelength!r}")
    if spacing <= 0:
        raise ValidationError(f"阵元间距必须为正数，收到 {spacing!r}")
    positions = (np.arange(m, dtype=np.float64) - (m - 1) / 2.0) * spacing
    if position_errors is not None:
        err = np.asarray(position_errors, dtype=np.float64).reshape(-1)
        if err.size != m:
            raise ValidationError(
                f"阵元位置误差长度必须等于阵元数 M={m}，收到长度 {err.size}"
            )
        positions = positions + err
    return positions


def steering_vector(
    angle_deg: float | np.ndarray, positions: np.ndarray, wavelength: float
) -> np.ndarray:
    """计算方向向量（可批量）。

    约定波程差为 ``positions * sin(theta)``，即
    ``a(theta)[m] = exp(j 2pi p_m sin(theta)/lambda)``。
    因此 ``theta -> -theta`` 等价于 ``a -> conj(a)``。
    单角度返回 ``(M,)``；角度数组返回 ``(M, len(angles))``。
    """
    angles = np.atleast_1d(np.asarray(angle_deg, dtype=np.float64))
    sin_theta = np.sin(np.deg2rad(angles))
    phase = (2.0 * math.pi / wavelength) * np.outer(positions, sin_theta)
    a = np.exp(1j * phase)
    if np.isscalar(angle_deg) or np.asarray(angle_deg).ndim == 0:
        return a[:, 0]
    return a


def has_grating_ambiguity(spacing: float, wavelength: float) -> bool:
    """间距大于半波长即存在栅瓣（空间混叠）。"""
    return spacing > wavelength / 2.0


def grating_alias_pairs(
    positions: np.ndarray,
    wavelength: float,
    angle_min: float = -90.0,
    angle_max: float = 90.0,
    angle_step: float = 0.5,
    tol_deg: float = 1e-6,
) -> list[tuple[float, float]]:
    """枚举扫描范围内会被阵列混淆的角度对。

    对扫描栅格上的每个角度 ``theta``，由混叠关系
    ``sin(theta') = sin(theta) +- k * lambda/d`` 反解候选 ``theta'``，
    保留落在扫描范围内、且与 ``theta`` 不同的代表对。

    理想均匀线阵（位置误差为 0）的混叠周期为 ``lambda/d``；非理想阵列
    一般没有严格栅瓣，此时返回空列表。
    """
    m = positions.size
    spacing_grid = positions[1:] - positions[:-1]
    ideal = np.arange(m, dtype=np.float64) - (m - 1) / 2.0
    d = wavelength / 2.0  # 兜底；非理想阵列不会用到
    if m >= 2:
        d = float(np.mean(spacing_grid))
    is_ideal_ula = bool(
        np.allclose(positions, ideal * d, atol=1e-9 * max(wavelength, 1.0))
    )
    if not is_ideal_ula or not has_grating_ambiguity(d, wavelength):
        return []

    aliases: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    kmax = int(math.ceil(2.0 / (wavelength / d))) + 1
    angles = np.arange(angle_min, angle_max + 1e-12, angle_step, dtype=np.float64)
    for theta in angles:
        s = math.sin(math.radians(float(theta)))
        for k in range(-kmax, kmax + 1):
            if k == 0:
                continue
            sp = s + k * wavelength / d
            if sp < -1.0 - tol_deg or sp > 1.0 + tol_deg:
                continue
            sp = min(1.0, max(-1.0, sp))
            phi = math.degrees(math.asin(sp))
            if phi < angle_min - tol_deg or phi > angle_max + tol_deg:
                continue
            if abs(phi - float(theta)) <= tol_deg:
                continue
            key = tuple(sorted((round(float(theta), 6), round(phi, 6))))
            if key not in seen:
                seen.add(key)  # type: ignore[arg-type]
                aliases.append((key[0], key[1]))  # type: ignore[index]
    return aliases

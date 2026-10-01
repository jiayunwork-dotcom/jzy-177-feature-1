"""滑动窗口 vs 指数遗忘：角度阶跃跟踪实验（文档取舍论证的数据来源）。

场景：单源在第 N0 条快拍后发生一次方位阶跃（theta0 -> theta1），
服务以两种增量方式持续维护协方差并周期性出 MUSIC 估计。

指标：
  * 跟踪时间：阶跃后估计进入新方向 ±0.5° 走廊并稳定不再离开所需的快拍数；
  * 稳态 RMSE：进入稳态后 200 个估计点相对真值的均方根误差（度）；
  * 遗忘参数取两档：lambda=0.95（时间常数 20）/0.98（50），
    窗口取 W=100/300。

用法：
    python -m experiments.tracking_experiment [--snr 15] [--k-per-side 1200]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from app.doa.engine import ArraySpec, EstConfig, estimate
from app.doa.synthesis import generate_snapshots
from app.sessions.accumulator import CovarianceAccumulator


def run(snr_db: float = 15.0, k_per_side: int = 1200, seed: int = 2026):
    m, spacing, wl = 8, 0.5, 1.0
    spec = ArraySpec(elements=m, spacing=spacing, wavelength=wl)
    theta0, theta1 = -30.0, 12.0
    rng_data_seed = seed

    s0 = generate_snapshots(m, k_per_side, [theta0], [1.0], snr_db,
                            spec.positions(), wl, seed=rng_data_seed)
    s1 = generate_snapshots(m, k_per_side, [theta1], [1.0], snr_db,
                            spec.positions(), wl, seed=rng_data_seed + 1)
    data = np.hstack([s0["data"], s1["data"]])  # 阶跃在 k_per_side 处
    step_at = k_per_side

    configs = [
        ("sliding W=100", "sliding", 100, None),
        ("sliding W=300", "sliding", 300, None),
        ("forgetting λ=0.95 (τ≈20)", "forgetting", None, 0.95),
        ("forgetting λ=0.98 (τ≈50)", "forgetting", None, 0.98),
    ]
    cfg = EstConfig(angle_step=0.5, source_count=1)
    out = []
    for name, mode, window, lam in configs:
        acc = CovarianceAccumulator(m, mode=mode, window_size=window,
                                    forgetting_factor=lam)
        angles: list[float | None] = []
        for k in range(data.shape[1]):
            acc.add_batch(data[:, k:k + 1])
            if acc.n_snapshots >= m:
                res = estimate(None, spec, cfg,
                               covariance=acc.covariance(),
                               n_snapshots=acc.n_snapshots)
                angles.append(float(res["angles"][0]))
            else:
                angles.append(None)

        # 跟踪时间：阶跃之后首次进入 ±0.5° 走廊，并在此后 20 拍内不离开
        track_k = None
        settle = 20
        for k in range(step_at, len(angles)):
            window_after = angles[k:k + settle]
            if all(a is not None and abs(a - theta1) <= 0.5 for a in window_after):
                track_k = k - step_at + 1
                break
        tail_start = step_at + min(600, k_per_side // 2)
        tail = [a for a in angles[tail_start:] if a is not None]
        rmse = float(np.sqrt(np.mean([(a - theta1) ** 2 for a in tail])))
        out.append({
            "method": name,
            "track_snapshots_to_0p5_deg": track_k,
            "steady_state_rmse_deg": round(rmse, 4),
        })
    report = {
        "scenario": {
            "snr_db": snr_db,
            "step": [theta0, theta1],
            "step_at_snapshot": step_at,
            "seed": seed,
            "criterion": "估计进入新方向 ±0.5° 走廊且连续 20 拍不离开",
        },
        "results": out,
    }
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, default=15.0)
    ap.add_argument("--k-per-side", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    print(json.dumps(run(args.snr, args.k_per_side, args.seed),
                     ensure_ascii=False, indent=2))

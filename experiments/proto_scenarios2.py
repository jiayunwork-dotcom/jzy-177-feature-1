"""原型2：中途进出 / 一闪而过 / 静止场景多种子精度统计。"""

import numpy as np

from app.doa.engine import ArraySpec, EstConfig, estimate
from app.doa.synthesis import generate_snapshots
from app.doa.array import steering_vector

M, SPACING, WL, K, STEP = 8, 0.5, 1.0, 50, 0.2
spec = ArraySpec(elements=M, spacing=SPACING, wavelength=WL)
cfg = EstConfig(angle_min=-90, angle_max=90, angle_step=STEP)


def est_batch(angles, powers, snr_db, seed):
    syn = generate_snapshots(
        m=M, k=K, angles_deg=angles, powers=powers, snr_db=snr_db,
        positions=spec.positions(), wavelength=WL, seed=seed,
    )
    R = (syn["data"] @ syn["data"].conj().T) / K
    res = estimate(None, spec, cfg, covariance=R, n_snapshots=K,
                   effective_snapshots=K)
    Rinv = np.linalg.pinv(R)
    a = steering_vector(np.asarray(res["angles"]), spec.positions(), WL)
    pmvdr = [1.0 / np.real(av.conj() @ Rinv @ av) for av in a.T]
    return res["mdl_source_count"], res["angles"], pmvdr, res["resolved"]


def main():
    print("== 中途进出：+50 源只在 b10..19 出现 ==")
    for b in range(30):
        ang = [-20.0, 25.0] + ([50.0] if 10 <= b <= 19 else [])
        pw = [1.0] * len(ang)
        mdl, est, pm, resolved = est_batch(ang, pw, 20.0, 3000 + b)
        print(f"b={b:02d} n={len(ang)} mdl={mdl} est={np.round(est, 2)} resolved={resolved}")

    print("== 一闪而过：仅 b5 混入第三源 ==")
    for b in range(10):
        ang = [-20.0, 25.0] + ([50.0] if b == 5 else [])
        mdl, est, pm, resolved = est_batch(ang, [1.0] * len(ang), 20.0, 4000 + b)
        print(f"b={b:02d} n={len(ang)} mdl={mdl} est={np.round(est, 2)} resolved={resolved}")

    print("== 静止双目标：多种子序列，统计 >0.1 度的批 ==")
    for base in range(0, 10):
        bad = []
        maxerr = 0.0
        for b in range(30):
            mdl, est, _, _ = est_batch([-20.0, 25.0], [1.0, 1.0], 20.0,
                                       base * 1000 + b)
            errs = [abs(est[0] + 20), abs(est[1] - 25)]
            maxerr = max(maxerr, max(errs))
            if mdl != 2 or max(errs) > 0.1:
                bad.append((b, mdl, np.round(est, 3)))
        print(f"seed_base={base}: bad={bad} maxerr={maxerr:.3f}")


if __name__ == "__main__":
    main()

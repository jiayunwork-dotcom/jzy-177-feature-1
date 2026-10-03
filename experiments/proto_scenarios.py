"""原型：按验收剧本逐批生成数据并出估计，观察交叉附近的峰数/角度/功率。"""

import numpy as np

from app.doa.engine import ArraySpec, EstConfig, estimate
from app.doa.synthesis import generate_snapshots
from app.doa.array import steering_vector

M, SPACING, WL, K, W, STEP = 8, 0.5, 1.0, 50, 50, 0.2


def run_scenario(name, angle_fn, powers_fn, snr_db, n_batches, seed=1000):
    spec = ArraySpec(elements=M, spacing=SPACING, wavelength=WL)
    cfg = EstConfig(angle_min=-90, angle_max=90, angle_step=STEP)
    R = None
    rows = []
    for b in range(n_batches):
        angles = angle_fn(b)
        powers = powers_fn(b)
        syn = generate_snapshots(
            m=M, k=K, angles_deg=angles, powers=powers, snr_db=snr_db,
            positions=spec.positions(), wavelength=WL, seed=seed + b,
        )
        X = syn["data"]
        R = (X @ X.conj().T) / K  # 窗口=批大小，每批协方差只取决于本批
        res = estimate(None, spec, cfg, covariance=R, n_snapshots=K,
                       effective_snapshots=K)
        # MVDR 逐角功率
        Rinv = np.linalg.pinv(R)
        a = steering_vector(np.asarray(res["angles"]), spec.positions(), WL)
        pmvdr = [1.0 / np.real(av.conj() @ Rinv @ av) for av in a.T]
        rows.append((b, angles, res["mdl_source_count"], res["angles"],
                     pmvdr, res["signal_power_est"]))
    print(f"===== {name} (SNR {snr_db}) =====")
    for b, truth, mdl, ang, pmvdr, peig in rows:
        flag = " <==" if mdl != len(truth) else ""
        print(f"b={b:02d} true={np.round(truth, 2)} mdl={mdl} "
              f"est={np.round(ang, 2)} mvdrP={np.round(pmvdr, 3)} "
              f"eigP={np.round(peig, 3)}{flag}")
    return rows


def main():
    # 静止双目标
    run_scenario("stationary", lambda b: [-20.0, 25.0], lambda b: [1.0, 1.0],
                 20.0, 30)

    # 交叉
    def cross_angles(b):
        t = b / 39.0
        return [-15.0 + 30.0 * t, 15.0 - 30.0 * t]

    run_scenario("crossing", cross_angles, lambda b: [1.0, 0.5], 15.0, 40,
                 seed=2026)


if __name__ == "__main__":
    main()

"""原型3：用 Tracker 跑全部验收剧本，检查编号、状态、交叉对号。"""

import numpy as np

from app.doa.engine import ArraySpec, EstConfig, estimate
from app.doa.synthesis import generate_snapshots
from app.doa.beamform import beamscan_powers
from app.tracker import Tracker

M, SPACING, WL, K, STEP = 8, 0.5, 1.0, 50, 0.2
spec = ArraySpec(elements=M, spacing=SPACING, wavelength=WL)
cfg = EstConfig(angle_min=-90, angle_max=90, angle_step=STEP)


def run(name, angle_fn, powers_fn, snr_db, n_batches, seed):
    tracker = Tracker()
    print(f"\n===== {name} =====")
    for b in range(n_batches):
        truth = angle_fn(b)
        powers = powers_fn(b)
        syn = generate_snapshots(
            m=M, k=K, angles_deg=truth, powers=powers, snr_db=snr_db,
            positions=spec.positions(), wavelength=WL, seed=seed,
        )
        # 验收设置：每批 50、窗口 50 -> R 只由本批决定
        R = (syn["data"] @ syn["data"].conj().T) / K
        R = (R + R.conj().T) * 0.5
        res = estimate(None, spec, cfg, covariance=R, n_snapshots=K,
                       effective_snapshots=K)
        bp = beamscan_powers(res["angles"], R, spec.positions(), WL)
        out = tracker.update(b + 1, res["angles"], bp)
        ev = " ".join(f"{e['type']}#{e.get('track_id', e.get('group_id'))}"
                      for e in out["events"])
        asg = ", ".join(
            f"{a['angle']:.1f}->T{a['track_id']}" if a["associated"]
            else f"{a['angle']:.1f}->-"
            for a in out["assignments"])
        print(f"b={b:02d} true={np.round(truth, 1)} [{asg}] {ev}")
    print("--- 最终航迹 ---")
    for t in tracker.list_tracks():
        print(f"T{t.track_id} {t.status} hits={t.hits} "
              f"first={t.first_version} last={t.last_hit_version} "
              f"firstA={t.history[0]['angle']:.2f} "
              f"lastA={t.history[-1]['angle']:.2f}")
    return tracker


# 静止
t1 = run("stationary", lambda b: [-20.0, 25.0], lambda b: [1.0, 1.0],
         20.0, 30, seed=100)

# 中途进出
t2 = run("in-out",
         lambda b: [-20.0, 25.0] + ([50.0] if 10 <= b <= 19 else []),
         lambda b: [1.0] * (3 if 10 <= b <= 19 else 2),
         20.0, 30, seed=3100)

# 一闪而过
t3 = run("flash",
         lambda b: [-20.0, 25.0] + ([50.0] if b == 5 else []),
         lambda b: [1.0] * (3 if b == 5 else 2),
         20.0, 12, seed=4000)

# 交叉
t4 = run("crossing",
         lambda b: [-15.0 + 30.0 * b / 39.0, 15.0 - 30.0 * b / 39.0],
         lambda b: [1.0, 0.5], 15.0, 40, seed=2026)

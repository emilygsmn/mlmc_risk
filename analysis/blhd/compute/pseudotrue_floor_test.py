"""Decisive test of whether the alpha=0.95 'bias floor' is real estimator bias or pseudo-true
reference noise. Builds a HIGH-ACCURACY Q_full (many fine samples pooled, SE ~0.02), re-runs the
BL-HD estimator over the cost grid saving every q_hat, and computes RMSE-vs-cost against BOTH the
accurate reference AND a single-300k-draw noisy reference. Same q_hats -> any floor difference is
purely the reference. Crash-safe (saves every budget). Detached long job.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import RESULTS_DIR
from mlmc_risk_estimation.blhd.samplers.portfolio import portfolio_sampler
from mlmc_risk_estimation.blhd.experiment import calibrate_domain, adapt_tuning_blhd_quantile
from mlmc_risk_estimation.blhd.baselines import order_statistic_quantile, harrell_davis

ALPHA = 0.95
N_ACC_TOTAL = 20_000_000      # pooled fine sample for accurate Q_full (SE ~0.024)
CHUNK = 1_000_000
N_NOISY = 300_000             # single-draw reference reproducing the old inflated floor
COST_GRID = list(np.geomspace(5e5, 1e7, 6))
N_TRIALS = 20
CF, CDG, CM, SMH, K, R = 210.0, 1.0, 0.70, 40.0, 20, 3
OUT = str(RESULTS_DIR / f"pseudotrue_floor_test_alpha{ALPHA}")


def main() -> None:
    """Build an accurate pooled pseudo-true VaR, re-run BL-HD over the cost grid, and compare RMSE
    against the accurate and noisy references."""
    s = portfolio_sampler()
    S0, S1, _ = calibrate_domain(s["fine"], n_pilot=200)

    print(f"Building accurate Q_full from {N_ACC_TOTAL:,} fine draws...", flush=True)
    pool = []
    for j in range(N_ACC_TOTAL // CHUNK):
        np.random.seed(500 + j)
        pool.append(s["fine"](CHUNK))
        if (j + 1) % 5 == 0:
            print(f"  {(j+1)*CHUNK:,}/{N_ACC_TOTAL:,} drawn", flush=True)
    pool = np.sort(np.concatenate(pool))
    Q_acc = float(order_statistic_quantile(pool, ALPHA))
    np.random.seed(99); Q_noisy = float(order_statistic_quantile(np.sort(s["fine"](N_NOISY)), ALPHA))
    print(f"\nQ_full accurate (N={N_ACC_TOTAL:,}) = {Q_acc:.4f}", flush=True)
    print(f"Q_full noisy   (N={N_NOISY:,})   = {Q_noisy:.4f}   (diff {Q_noisy-Q_acc:+.4f})\n", flush=True)

    rows = []
    for C in COST_GRID:
        qh = []
        for t in range(N_TRIALS):
            rng = np.random.default_rng(7000 + t)
            q, _ = adapt_tuning_blhd_quantile(ALPHA, C, s["coarse"], s["level0"], s["level1"],
                                              CF, CDG, CM, S0, S1, R, SMH, K=K, rng=rng, verbose=False)
            qh.append(q)
        qh = np.array(qh)
        rmse_acc = float(np.sqrt(np.mean((qh - Q_acc) ** 2)))
        rmse_noisy = float(np.sqrt(np.mean((qh - Q_noisy) ** 2)))
        std = float(qh.std(ddof=1))
        rows.append(dict(C=C, C_over_cf=C / CF, rmse_accurate=rmse_acc, rmse_noisy=rmse_noisy,
                         std_qhat=std, mean_qhat=float(qh.mean()), qhats=qh.tolist()))
        print(f"C/c_f={C/CF:7.0f}  RMSE(accurate)={rmse_acc:.4f}  RMSE(noisy)={rmse_noisy:.4f}  "
              f"std(q_hat)={std:.4f}", flush=True)
        json.dump(dict(alpha=ALPHA, Q_acc=Q_acc, Q_noisy=Q_noisy, N_acc=N_ACC_TOTAL,
                       n_trials=N_TRIALS, rows=rows), open(OUT + ".json", "w"), indent=2)
    print(f"\nSaved {OUT}.json\nALL DONE", flush=True)


if __name__ == "__main__":
    main()

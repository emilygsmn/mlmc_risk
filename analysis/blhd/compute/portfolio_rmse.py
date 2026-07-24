"""Compute RMSE-vs-cost for BL-HD on the EIOPA-benchmark portfolio (full revaluation as the fine
model, delta-gamma as the coarse surrogate), against the HD/OS fine-only baselines, for several VaR
confidence levels alpha. Ground truth is a large-sample full-revaluation pseudo-true VaR (no closed
form exists). Costs c_f/c_dg/c_m are the benchmarked values (mlmc_risk_estimation.blhd.cost.
measure_costs). One JSON+CSV per alpha is written to results/blhd/; plot with
analysis/blhd/plot/rmse_vs_cost.py.

The sampler and CDF/PDF domain [S_0, S_1] do not depend on alpha, so they are built ONCE and reused
across the alpha loop; only the pseudo-true VaR is recomputed per alpha (different loss tail).

KNOWN ISSUE: run_rmse_vs_cost_portfolio is sequential and can leak memory across trials, so at large
n_trials it may be OOM-killed on the largest cost point. Requires network access (historical-data
download).
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import RESULTS_DIR

from mlmc_risk_estimation.blhd.samplers.portfolio import portfolio_sampler
from mlmc_risk_estimation.blhd.experiment import (run_rmse_vs_cost_portfolio, calibrate_domain,
                                                  estimate_pseudo_true_quantile, save_results)
from mlmc_risk_estimation.blhd.plotting import build_filename_tag, cost_grid_tag

ALPHAS = (0.95, 0.99, 0.995)


def _pseudo_true(sampler: dict, alpha: float, n_fallback: int, seed: int) -> float:
    """Accurate pseudo-true VaR from the cached large-pool reference (build_accurate_pseudotrue.py)
    if available; else fall back to a single n_fallback-sample order-statistic quantile. The cache
    matters: a single-300k reference has SE ~0.2 -- the size of the RMSE floor -- so measuring
    against it inflates the apparent bias. See build_accurate_pseudotrue.py."""
    cache = RESULTS_DIR / "pseudotrue_accurate.json"
    if cache.exists():
        d = json.load(open(cache))
        key = f"{alpha}"
        if key in d.get("Q", {}):
            print(f"  [pseudo-true] using cached accurate Q_full (N={d['n_total']:,}, "
                  f"SE~{d['Q_se'][key]:.4f})")
            return float(d["Q"][key])
    print(f"  [pseudo-true] WARNING: no cache; falling back to single {n_fallback:,}-sample quantile "
          "(noisy). Run build_accurate_pseudotrue.py first.")
    return estimate_pseudo_true_quantile(sampler["fine"], alpha, n_fallback, seed=seed)


def main() -> None:
    """Run the portfolio RMSE-vs-cost sweep for every alpha in ALPHAS and save the results."""
    r = 3
    c_f, c_dg, c_m, s_mh = 210.0, 1.0, 0.70, 40.0
    n_trials, K = 50, 20
    n_pseudo_true, n_pilot, seed = 300_000, 200, 0
    # Range sized from a measured probe (this runner is single-core -- the portfolio sampler cannot
    # cross a process boundary, so there is no worker pool). Memory is NOT the limit: at fixed
    # C=1e7 peak RSS is FLAT across trials (~3.1 GB), so the "cross-trial leak" the docstring warns
    # about is really working set, and 50 trials is safe. Wall time is the limit: ~12-20s per BL-HD
    # trial at C=1e7. Top end 1e7 (C/c_f ~ 47,600) is the largest budget giving a tolerable ~75-90
    # min run for all 3 alphas at 50 trials; 2.5e7 (~37s/trial) would roughly triple that. Low end
    # 5e5 keeps the pilot MH chain length N_m_p = C_star/(10_000*c_m) ~ 71 (>0). One JSON is saved
    # per alpha as it completes, so a mid-run crash only loses the in-progress alpha.
    cost_grid = list(np.geomspace(5e5, 1e7, 6))

    sampler = portfolio_sampler()
    S_0, S_1, _ = calibrate_domain(sampler["fine"], n_pilot=n_pilot)
    print(f"Domain [S_0,S_1]=[{S_0:.2f},{S_1:.2f}]  cost_grid={[f'{c:,.0f}' for c in cost_grid]}\n")

    for alpha in ALPHAS:
        true_q = _pseudo_true(sampler, alpha, n_pseudo_true, seed)
        print(f"{"=" * 70}\nalpha={alpha}  pseudo-true VaR={true_q:.6f}\n{"=" * 70}")

        results = run_rmse_vs_cost_portfolio(sampler, alpha, cost_grid, S_0, S_1, r, c_f, c_dg, c_m,
                                             s_mh, true_q, n_trials=n_trials, K=K, seed=seed,
                                             verbose=True)

        tag = build_filename_tag(model="portfolio", alpha=alpha, cf=c_f, cdg=c_dg, cm=c_m, smh=s_mh,
                                 ntrials=n_trials, K=K)
        # Cost-grid range appended so re-runs over a DIFFERENT grid don't overwrite each other.
        tag = f"{tag}_{cost_grid_tag(cost_grid)}"
        meta = dict(model="portfolio", alpha=alpha, S_0=S_0, S_1=S_1, r=r, c_f=c_f, c_dg=c_dg,
                    c_m=c_m, s_mh=s_mh, n_trials=n_trials, K=K, n_pseudo_true=n_pseudo_true,
                    true_q=true_q, tag=tag, plot_labels={"BL-HD": "BL-HD"})

        json_path, csv_path = save_results(results, meta, str(RESULTS_DIR), f"rmse_vs_cost_{tag}")
        print(f"Saved {json_path}\n      {csv_path}\n")

    print("All alpha runs complete.")


if __name__ == "__main__":
    main()

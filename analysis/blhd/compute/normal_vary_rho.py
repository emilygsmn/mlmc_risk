"""Compute RMSE-vs-cost for BL-HD across several fine/DG correlations rho (bivariate-normal model),
against the HD/OS fine-only baselines. Writes results + meta to results/blhd/ as JSON+CSV; plot with
analysis/blhd/plot/rmse_vs_cost.py.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import RESULTS_DIR

from mlmc_risk_estimation.blhd.experiment import run_rmse_vs_cost, draw_normal, save_results
from mlmc_risk_estimation.blhd.samplers.analytical import normal_sampler
from mlmc_risk_estimation.blhd.plotting import build_filename_tag, cost_grid_tag


def main() -> None:
    """Run RMSE-vs-cost for BL-HD across rho_list at a fixed alpha."""
    mu, sd = 0.0, 1.0
    # Seven rho values spread across the range, with extra resolution around rho ~ 0.8-0.95 where
    # BL-HD crosses the HD fine-only baseline (the break-even correlation). Plot with
    # analysis/blhd/plot/rmse_vs_rho.py.
    rho_list = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    alpha = 0.995
    S_0, S_1, r = 0.0, 4.0, 3
    c_f, c_dg, c_m, s_mh = 250.0, 1.0, 0.05, 0.05
    n_trials, K = 50, 20
    # Budgets chosen as round decades of C/c_f (= 100, 1000, 10000, 100000) so the per-budget curves
    # in the RMSE-vs-rho view carry legible labels instead of geomspace artefacts like 6,604.
    cost_grid = [c_f * ratio for ratio in (100, 1000, 10000, 100000)]

    true_q = float(norm.ppf(alpha, mu, sd))
    est = dict(c_f=c_f, c_dg=c_dg, c_m=c_m, S_0=S_0, S_1=S_1, r=r, s_mh=s_mh, K=K)
    bl_arms = [(f"BL-HD (rho={rho:g})", normal_sampler, (rho, mu, sd), est)
               for rho in sorted(rho_list)]
    baseline = (draw_normal, (mu, sd), c_f)

    results = run_rmse_vs_cost(bl_arms, baseline, cost_grid, alpha, true_q,
                               n_trials=n_trials, seed=0, verbose=True)

    tag = build_filename_tag(model="normal", mu=mu, sd=sd, rho=rho_list, alpha=alpha,
                             cf=c_f, cdg=c_dg, cm=c_m, smh=s_mh, ntrials=n_trials, K=K)
    # Cost-grid range appended so re-runs over a DIFFERENT grid don't overwrite each other.
    tag = f"{tag}_{cost_grid_tag(cost_grid)}"
    plot_labels = {f"BL-HD (rho={rho:g})": rf"BL-HD ($\varrho={rho:g}$)" for rho in sorted(rho_list)}
    meta = dict(model="normal", mu=mu, sd=sd, rho_list=rho_list, alpha=alpha, S_0=S_0, S_1=S_1, r=r,
                c_f=c_f, c_dg=c_dg, c_m=c_m, s_mh=s_mh, n_trials=n_trials, K=K,
                true_q=true_q, tag=tag, plot_labels=plot_labels)

    json_path, csv_path = save_results(results, meta, str(RESULTS_DIR), f"rmse_vs_cost_{tag}")
    print(f"\nSaved {json_path}\n      {csv_path}")


if __name__ == "__main__":
    main()

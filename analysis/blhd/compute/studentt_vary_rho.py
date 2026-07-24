"""Compute RMSE-vs-cost for BL-HD across several fine/DG correlations rho (bivariate Student-t
model, dof degrees of freedom), against the HD/OS fine-only baselines. Writes results + meta to
results/blhd/ as JSON+CSV; plot with analysis/blhd/plot/rmse_vs_cost.py.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as scipy_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import RESULTS_DIR

from mlmc_risk_estimation.blhd.experiment import run_rmse_vs_cost, draw_studentt, save_results
from mlmc_risk_estimation.blhd.samplers.analytical import studentt_sampler
from mlmc_risk_estimation.blhd.plotting import build_filename_tag, cost_grid_tag


def main() -> None:
    """Run the Student-t RMSE-vs-cost sweep across rho at a single dof and save the results."""
    dof = 5
    rho_list = [0.5, 0.9, 0.99, 0.999]
    alpha = 0.995
    S_0, S_1, r = -5.0, 15.0, 3
    c_f, c_dg, c_m, s_mh = 250.0, 1.0, 0.05, 0.05
    n_trials, K = 50, 20
    cost_grid = list(np.geomspace(5e3, 5e6, 7))

    true_q = float(scipy_t.ppf(alpha, dof))
    est = dict(c_f=c_f, c_dg=c_dg, c_m=c_m, S_0=S_0, S_1=S_1, r=r, s_mh=s_mh, K=K)
    bl_arms = [(f"BL-HD (rho={rho:g})", studentt_sampler, (rho, dof), est)
               for rho in sorted(rho_list)]
    baseline = (draw_studentt, (dof,), c_f)

    results = run_rmse_vs_cost(bl_arms, baseline, cost_grid, alpha, true_q,
                               n_trials=n_trials, seed=0, verbose=True)

    tag = build_filename_tag(model="studentt", dof=dof, rho=rho_list, alpha=alpha,
                             cf=c_f, cdg=c_dg, cm=c_m, smh=s_mh, ntrials=n_trials, K=K)
    # Cost-grid range appended so re-runs over a DIFFERENT grid don't overwrite each other.
    tag = f"{tag}_{cost_grid_tag(cost_grid)}"
    plot_labels = {f"BL-HD (rho={rho:g})": rf"BL-HD ($\varrho={rho:g}$)" for rho in sorted(rho_list)}
    meta = dict(model="studentt", dof=dof, rho_list=rho_list, alpha=alpha, S_0=S_0, S_1=S_1, r=r,
                c_f=c_f, c_dg=c_dg, c_m=c_m, s_mh=s_mh, n_trials=n_trials, K=K,
                true_q=true_q, tag=tag, plot_labels=plot_labels)

    json_path, csv_path = save_results(results, meta, str(RESULTS_DIR), f"rmse_vs_cost_{tag}")
    print(f"\nSaved {json_path}\n      {csv_path}")


if __name__ == "__main__":
    main()

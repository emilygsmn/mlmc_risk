"""Bias^2/variance decomposition of RMSE^2 vs. cost (bivariate-normal model, fixed rho): BL-HD vs.
the HD/OS fine-only baselines, at equal total cost. Checks whether both error sources are actually
shrinking with cost, rather than one term stalling while the other keeps improving -- a direct
sanity check on whether BL-HD's internal error-budget split is well-balanced. Set N_BOOT=0 for
point estimates only (skips the bootstrap SEs). Writes results + meta to results/blhd/ as
JSON+CSV; plot with analysis/blhd/plot/bias_variance.py.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import RESULTS_DIR

from mlmc_risk_estimation.blhd.experiment import run_bias_variance_vs_cost, draw_normal, save_results
from mlmc_risk_estimation.blhd.samplers.analytical import normal_sampler
from mlmc_risk_estimation.blhd.plotting import build_filename_tag, cost_grid_tag


def main() -> None:
    """Run the bias/variance-vs-cost decomposition for BL-HD on the bivariate-normal model."""
    mu, sd, rho = 0.0, 1.0, 0.999
    alpha = 0.99
    S_0, S_1, r = 0.0, 4.0, 3
    c_f, c_dg, c_m, s_mh = 250.0, 1.0, 0.05, 0.05
    n_trials, K = 25, 20
    n_boot = 2000  # set to 0 for point estimates only (no bootstrap SEs)
    cost_grid = list(np.geomspace(3e5, 3e7, 7))

    true_q = float(norm.ppf(alpha, mu, sd))
    est = dict(c_f=c_f, c_dg=c_dg, c_m=c_m, S_0=S_0, S_1=S_1, r=r, s_mh=s_mh, K=K)
    bl_arms = [("BL-HD", normal_sampler, (rho, mu, sd), est)]
    baseline = (draw_normal, (mu, sd), c_f)

    results = run_bias_variance_vs_cost(bl_arms, baseline, cost_grid, alpha, true_q,
                                        n_trials=n_trials, seed=0, verbose=True, n_boot=n_boot)

    tag = build_filename_tag(model="normal", mu=mu, sd=sd, rho=rho, alpha=alpha, cf=c_f, cdg=c_dg,
                             cm=c_m, smh=s_mh, ntrials=n_trials, K=K, nboot=n_boot)
    # Cost-grid range appended so re-runs over a DIFFERENT grid don't overwrite each other.
    tag = f"{tag}_{cost_grid_tag(cost_grid)}"
    meta = dict(model="normal", mu=mu, sd=sd, rho=rho, alpha=alpha, S_0=S_0, S_1=S_1, r=r,
                c_f=c_f, c_dg=c_dg, c_m=c_m, s_mh=s_mh, n_trials=n_trials, K=K, n_boot=n_boot,
                true_q=true_q, tag=tag)

    json_path, csv_path = save_results(results, meta, str(RESULTS_DIR), f"bias_variance_{tag}")
    print(f"\nSaved {json_path}\n      {csv_path}")


if __name__ == "__main__":
    main()

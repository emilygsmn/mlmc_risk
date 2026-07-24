"""Classic single-arm RMSE-vs-cost comparison at equal total cost (bivariate-normal model, fixed
fine/DG correlation rho): BL-HD vs. the HD/OS fine-only baselines. The base case underlying the
vary-rho/vary-cost-ratio families. Writes results + meta to results/blhd/ as JSON+CSV; plot with
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


ALPHAS = (0.95, 0.99, 0.995)


def main() -> None:
    """Run the RMSE-vs-cost comparison for BL-HD vs. HD/OS across ALPHAS."""
    # rho=0.999 roughly matches the additive-noise test case's implied correlation
    # (sd/sqrt(sd**2+noise_sd**2) = 1/sqrt(1+0.05**2) ~ 0.9987 at noise_sd=0.05).
    mu, sd, rho = 0.0, 1.0, 0.999
    S_0, S_1, r = 0.0, 4.0, 3
    c_f, c_dg, c_m, s_mh = 250.0, 1.0, 0.05, 0.05
    n_trials, K = 50, 20
    cost_grid = list(np.geomspace(5e3, 5e7, 7))

    for alpha in ALPHAS:
        print(f"{"=" * 70}\nalpha={alpha}\n{"=" * 70}")
        true_q = float(norm.ppf(alpha, mu, sd))
        est = dict(c_f=c_f, c_dg=c_dg, c_m=c_m, S_0=S_0, S_1=S_1, r=r, s_mh=s_mh, K=K)
        bl_arms = [("BL-HD", normal_sampler, (rho, mu, sd), est)]
        baseline = (draw_normal, (mu, sd), c_f)

        results = run_rmse_vs_cost(bl_arms, baseline, cost_grid, alpha, true_q,
                                   n_trials=n_trials, seed=0, verbose=True)

        tag = build_filename_tag(model="normal", mu=mu, sd=sd, rho=rho, alpha=alpha,
                                 cf=c_f, cdg=c_dg, cm=c_m, smh=s_mh, ntrials=n_trials, K=K)
        # Cost-grid range appended so re-runs over a DIFFERENT grid don't overwrite each other.
        tag = f"{tag}_{cost_grid_tag(cost_grid)}"
        meta = dict(model="normal", mu=mu, sd=sd, rho=rho, alpha=alpha, S_0=S_0, S_1=S_1, r=r,
                    c_f=c_f, c_dg=c_dg, c_m=c_m, s_mh=s_mh, n_trials=n_trials, K=K,
                    true_q=true_q, tag=tag, plot_labels={"BL-HD": "BL-HD"})

        json_path, csv_path = save_results(results, meta, str(RESULTS_DIR), f"rmse_vs_cost_{tag}")
        print(f"Saved {json_path}\n      {csv_path}\n")

    print("All alpha runs complete.")


if __name__ == "__main__":
    main()

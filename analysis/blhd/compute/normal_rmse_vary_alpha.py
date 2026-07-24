"""RMSE-vs-cost for BL-HD on the bivariate-normal model at a FIXED correlation rho=0.99, swept over
several VaR levels alpha, to show how the estimator's cost curve shifts as the target quantile moves
into the tail. All alphas share one cost grid and are written to a SINGLE combined JSON so the plot
overlays them; each alpha carries its OWN HD fine-only baseline (the HD RMSE depends on alpha because
the target quantile does), so the figure has one BL-HD curve AND one HD reference per alpha.

Combined-JSON key convention (so the dedicated plotter can pick the arms apart):
  "BL-HD (alpha=<a>)" -> {cost, rmse, se}      (this project's estimator, black family)
  "HD (alpha=<a>)"    -> {cost, rmse, se}      (fine-only baseline at the same alpha)

Plot with analysis/blhd/plot/rmse_vs_cost_vary_alpha.py.
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

# Four alphas -> four BL-HD curves and four HD reference lines in the combined figure.
ALPHAS = (0.9, 0.95, 0.99, 0.995)


def main() -> None:
    """Run RMSE-vs-cost for BL-HD at fixed rho, swept over ALPHAS, into one combined JSON."""
    mu, sd, rho = 0.0, 1.0, 0.99
    S_0, S_1, r = 0.0, 4.0, 3
    c_f, c_dg, c_m, s_mh = 250.0, 1.0, 0.05, 0.05
    n_trials, K = 50, 20
    cost_grid = list(np.geomspace(5e3, 5e7, 8))

    combined = {}
    for alpha in ALPHAS:
        print(f"{"=" * 70}\nalpha={alpha}\n{"=" * 70}")
        true_q = float(norm.ppf(alpha, mu, sd))
        est = dict(c_f=c_f, c_dg=c_dg, c_m=c_m, S_0=S_0, S_1=S_1, r=r, s_mh=s_mh, K=K)
        bl_arms = [("BL-HD", normal_sampler, (rho, mu, sd), est)]
        baseline = (draw_normal, (mu, sd), c_f)

        res = run_rmse_vs_cost(bl_arms, baseline, cost_grid, alpha, true_q,
                               n_trials=n_trials, seed=0, verbose=True)
        # Re-key under the alpha so all four alphas coexist in one results dict.
        combined[f"BL-HD (alpha={alpha:g})"] = res["BL-HD"]
        combined[f"HD (alpha={alpha:g})"] = res["HD (fine)"]

    tag = build_filename_tag(model="normal", mu=mu, sd=sd, rho=rho, alpha=list(ALPHAS),
                             cf=c_f, cdg=c_dg, cm=c_m, smh=s_mh, ntrials=n_trials, K=K)
    tag = f"{tag}_{cost_grid_tag(cost_grid)}"
    # Explicit per-arm LaTeX labels: BL-HD arms name their alpha; HD arms carry the alpha too so the
    # plotter can pair each HD reference with its BL-HD curve by shade.
    plot_labels = {}
    for a in ALPHAS:
        plot_labels[f"BL-HD (alpha={a:g})"] = rf"BL-HD ($\alpha={a:g}$)"
        plot_labels[f"HD (alpha={a:g})"] = rf"HD ($\alpha={a:g}$)"
    meta = dict(model="normal", mu=mu, sd=sd, rho=rho, alphas=list(ALPHAS), S_0=S_0, S_1=S_1, r=r,
                c_f=c_f, c_dg=c_dg, c_m=c_m, s_mh=s_mh, n_trials=n_trials, K=K,
                tag=tag, plot_labels=plot_labels)

    json_path, csv_path = save_results(combined, meta, str(RESULTS_DIR),
                                       f"rmse_vs_cost_varyalpha_{tag}")
    print(f"\nSaved {json_path}\n      {csv_path}")


if __name__ == "__main__":
    main()

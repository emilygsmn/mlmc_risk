"""QUICK single-alpha (0.99) portfolio RMSE-vs-cost on CURRENT data, sized to finish in ~10 min:
trimmed n_trials / cost grid / pseudo-true vs the full portfolio_rmse.py. Produces the same JSON
schema (BL-HD + HD/OS fine baselines), so analysis/blhd/plot/rmse_vs_cost.py plots it unchanged.
Meant for a fast "does BL-HD beat HD" check on the refreshed 2026 data, not thesis-final numbers.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import RESULTS_DIR

from mlmc_risk_estimation.blhd.samplers.portfolio import portfolio_sampler
from mlmc_risk_estimation.blhd.experiment import (run_rmse_vs_cost_portfolio, calibrate_domain,
                                                  estimate_pseudo_true_quantile, save_results)
from mlmc_risk_estimation.blhd.plotting import build_filename_tag, cost_grid_tag

ALPHA = 0.99


def main() -> None:
    """Run a trimmed single-alpha portfolio RMSE-vs-cost sweep and save the results."""
    r = 3
    c_f, c_dg, c_m, s_mh = 210.0, 1.0, 0.70, 40.0
    n_trials, K = 30, 20
    n_pseudo_true, n_pilot, seed = 100_000, 200, 0
    cost_grid = list(np.geomspace(1e5, 3e6, 6))   # trimmed top budget keeps HD-baseline N_fine small

    sampler = portfolio_sampler()
    S_0, S_1, _ = calibrate_domain(sampler["fine"], n_pilot=n_pilot)
    true_q = estimate_pseudo_true_quantile(sampler["fine"], ALPHA, n_pseudo_true, seed=seed)
    print(f"Domain [{S_0:.1f},{S_1:.1f}]  pseudo-true VaR={true_q:.4f}  "
          f"cost_grid={[f'{c:,.0f}' for c in cost_grid]}\n", flush=True)

    results = run_rmse_vs_cost_portfolio(sampler, ALPHA, cost_grid, S_0, S_1, r, c_f, c_dg, c_m,
                                         s_mh, true_q, n_trials=n_trials, K=K, seed=seed,
                                         verbose=True)

    tag = build_filename_tag(model="portfolioQUICK", alpha=ALPHA, cf=c_f, cdg=c_dg, cm=c_m,
                             smh=s_mh, ntrials=n_trials, K=K)
    tag = f"{tag}_{cost_grid_tag(cost_grid)}"
    meta = dict(model="portfolio", alpha=ALPHA, S_0=S_0, S_1=S_1, r=r, c_f=c_f, c_dg=c_dg, c_m=c_m,
                s_mh=s_mh, n_trials=n_trials, K=K, n_pseudo_true=n_pseudo_true,
                true_q=true_q, tag=tag, plot_labels={"BL-HD": "BL-HD"})

    json_path, csv_path = save_results(results, meta, str(RESULTS_DIR), f"rmse_vs_cost_{tag}")
    print(f"\nSaved {json_path}")


if __name__ == "__main__":
    main()

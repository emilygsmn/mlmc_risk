"""RMSE-vs-cost for BL-HD on the bivariate Student-t model at alpha=0.995, swept over the coarse/fine
correlation rho, for THREE tail heaviness settings dof in {5, 10, 15}. One JSON+CSV per dof (same
schema as studentt_vary_rho.py, so analysis/blhd/plot/rmse_vs_cost.py and rmse_vs_rho.py both work),
so the three tail regimes can be compared side by side.

Cost span: geomspace(5e3, 1.5e7, 8) -- the widest budget range that stays in the parallel driver's
proven-safe memory envelope (the analytical normal sweeps ran to ~3e7 in parallel; the portfolio
bias-variance ran to 1.5e7; 1.5e7 here keeps 8 concurrent Student-t trials well clear of OOM while
still ~3x wider than the old 5e6 grid). n_trials=50. Domains reuse the calibrated [S_0,S_1] from the
existing per-dof runs. RESUMABLE: a dof whose output JSON already exists is skipped.
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

# dof -> calibrated fine-model domain [S_0, S_1] (from the existing per-dof studentt runs).
DOF_DOMAIN = {
    5:  (-5.0, 15.0),
    10: (-3.930010276843804, 11.790030830531412),
    15: (-3.6540282617732025, 10.962084785319608),
}


def main() -> None:
    """Run the Student-t RMSE-vs-cost sweep for each dof in DOF_DOMAIN, skipping already-saved
    results (resumable)."""
    rho_list = [0.5, 0.9, 0.99, 0.999]
    alpha = 0.995
    r = 3
    c_f, c_dg, c_m, s_mh = 250.0, 1.0, 0.05, 0.05
    n_trials, K = 50, 20
    cost_grid = list(np.geomspace(5e3, 1.5e7, 8))

    for dof, (S_0, S_1) in DOF_DOMAIN.items():
        tag = build_filename_tag(model="studentt", dof=dof, rho=rho_list, alpha=alpha,
                                 cf=c_f, cdg=c_dg, cm=c_m, smh=s_mh, ntrials=n_trials, K=K)
        tag = f"{tag}_{cost_grid_tag(cost_grid)}"
        out_json = RESULTS_DIR / f"rmse_vs_cost_{tag}.json"
        if out_json.exists():
            print(f"dof={dof}: already done ({out_json.name}) -- skipping (resume).\n")
            continue

        print(f"{"=" * 70}\ndof={dof}  domain=[{S_0:.3f},{S_1:.3f}]  "
              f"cost_grid={[f'{c:,.0f}' for c in cost_grid]}\n{"=" * 70}")
        true_q = float(scipy_t.ppf(alpha, dof))
        est = dict(c_f=c_f, c_dg=c_dg, c_m=c_m, S_0=S_0, S_1=S_1, r=r, s_mh=s_mh, K=K)
        bl_arms = [(f"BL-HD (rho={rho:g})", studentt_sampler, (rho, dof), est)
                   for rho in sorted(rho_list)]
        baseline = (draw_studentt, (dof,), c_f)

        results = run_rmse_vs_cost(bl_arms, baseline, cost_grid, alpha, true_q,
                                   n_trials=n_trials, seed=0, verbose=True)

        plot_labels = {f"BL-HD (rho={rho:g})": rf"BL-HD ($\varrho={rho:g}$)"
                       for rho in sorted(rho_list)}
        meta = dict(model="studentt", dof=dof, rho_list=rho_list, alpha=alpha, S_0=S_0, S_1=S_1, r=r,
                    c_f=c_f, c_dg=c_dg, c_m=c_m, s_mh=s_mh, n_trials=n_trials, K=K,
                    true_q=true_q, tag=tag, plot_labels=plot_labels)

        json_path, csv_path = save_results(results, meta, str(RESULTS_DIR), f"rmse_vs_cost_{tag}")
        print(f"Saved {json_path}\n      {csv_path}\n")

    print("All dof runs complete.")


if __name__ == "__main__":
    main()

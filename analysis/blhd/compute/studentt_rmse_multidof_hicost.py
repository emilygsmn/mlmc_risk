"""High-cost EXTENSION of studentt_rmse_multidof.py: same Student-t RMSE-vs-cost sweep (dof 5/10/15,
alpha=0.995, rho in {0.5,0.9,0.99,0.999}, n_trials=50) with ONE further budget appended by continuing
the base grid's exact log-spacing (top 1.5e7 -> ~4.71e7, C/c_f ~ 188,000), so the lower-rho BL-HD
curves get a further step toward their asymptotic O(C^-1/2) regime while the point stays regularly
spaced on the plot.

Memory: this box has only ~9 GB RAM, and peak use ~ n_workers * (top-budget coarse arrays). The
throttle to N_WORKERS=2 keeps two concurrent ~4.71e7 trials (~2.4 GB each) plus system under 9 GB.
This is the ceiling: the NEXT regular step (~1.48e8) would need ~4.7 GB/trial and OOM the box.
RESUMABLE: a dof whose output JSON already exists is skipped.
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

DOF_DOMAIN = {
    5:  (-5.0, 15.0),
    10: (-3.930010276843804, 11.790030830531412),
    15: (-3.6540282617732025, 10.962084785319608),
}

N_WORKERS = 2   # throttled for the 9 GB box; peak memory ~ N_WORKERS * top-budget arrays


def main() -> None:
    """Run the high-cost-extension Student-t RMSE-vs-cost sweep for each dof in DOF_DOMAIN, skipping
    already-saved results (resumable)."""
    rho_list = [0.5, 0.9, 0.99, 0.999]
    alpha = 0.995
    r = 3
    c_f, c_dg, c_m, s_mh = 250.0, 1.0, 0.05, 0.05
    n_trials, K = 50, 20
    # Extend the base grid (geomspace 5e3..1.5e7, 8 pts) by CONTINUING its exact log-spacing, so the
    # added budget sits regularly among the others on the plot rather than at an odd gap. The base
    # ratio is ~3.14x/step; one further step gives ~4.71e7 (C/c_f ~ 188,000). Only ONE point is added:
    # the next regular step (~1.48e8) needs ~4.7 GB per trial x 2 workers, which exceeds this 9 GB box.
    base = np.geomspace(5e3, 1.5e7, 8)
    ratio = base[1] / base[0]
    cost_grid = list(base) + [float(base[-1] * ratio)]

    for dof, (S_0, S_1) in DOF_DOMAIN.items():
        tag = build_filename_tag(model="studentt", dof=dof, rho=rho_list, alpha=alpha,
                                 cf=c_f, cdg=c_dg, cm=c_m, smh=s_mh, ntrials=n_trials, K=K)
        tag = f"{tag}_{cost_grid_tag(cost_grid)}"
        out_json = RESULTS_DIR / f"rmse_vs_cost_{tag}.json"
        if out_json.exists():
            print(f"dof={dof}: already done ({out_json.name}) -- skipping (resume).\n")
            continue

        print(f"{"=" * 70}\ndof={dof}  domain=[{S_0:.3f},{S_1:.3f}]  N_WORKERS={N_WORKERS}\n"
              f"cost_grid={[f'{c:,.0f}' for c in cost_grid]}\n{"=" * 70}")
        true_q = float(scipy_t.ppf(alpha, dof))
        est = dict(c_f=c_f, c_dg=c_dg, c_m=c_m, S_0=S_0, S_1=S_1, r=r, s_mh=s_mh, K=K)
        bl_arms = [(f"BL-HD (rho={rho:g})", studentt_sampler, (rho, dof), est)
                   for rho in sorted(rho_list)]
        baseline = (draw_studentt, (dof,), c_f)

        results = run_rmse_vs_cost(bl_arms, baseline, cost_grid, alpha, true_q,
                                   n_trials=n_trials, seed=0, n_workers=N_WORKERS, verbose=True)

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

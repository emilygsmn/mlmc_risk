"""Runs the studentt_vary_rho.py comparison separately for dof in {3, 5, 10} -- everything
identical across all three EXCEPT the domain [S_0, S_1], which is scaled PROPORTIONALLY to each
dof's own t(dof).ppf(alpha) quantile (using the ratio validated for dof=5: S_0=-1.240*q,
S_1=3.720*q). A single fixed domain across all three dof would give some of them disproportionately
more "wasted" grid resolution than others (dof=10's much lighter tail vs. dof=3's much heavier one),
confounding any performance comparison between them -- proportional scaling keeps the comparison
fair, each dof getting the same RELATIVE headroom around its own target quantile.

Each dof's results are saved to their own JSON+CSV file (dof baked into the tag), independent of
the other runs. Plot each with analysis/blhd/plot/rmse_vs_cost.py.
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

# Ratios derived from the dof=5 domain (S_0=-5, S_1=15) validated in studentt_vary_rho.py, relative
# to t(dof=5).ppf(0.995) = 4.032.
_RATIO_S0 = -5.0 / scipy_t.ppf(0.995, df=5)
_RATIO_S1 = 15.0 / scipy_t.ppf(0.995, df=5)


def domain_for_dof(dof: int, alpha: float = 0.995) -> tuple[float, float]:
    """Proportionally-scaled S_0, S_1 for a given dof, using the same relative headroom validated
    for dof=5."""
    q = scipy_t.ppf(alpha, df=dof)
    return _RATIO_S0 * q, _RATIO_S1 * q


def main() -> None:
    """Run the Student-t RMSE-vs-cost sweep for dof in {5, 10, 15} and save the results."""
    rho_list = [0.5, 0.9, 0.99, 0.999]
    alpha = 0.995
    r = 3
    c_f, c_dg, c_m, s_mh = 250.0, 1.0, 0.05, 0.05
    n_trials, K = 50, 20
    cost_grid = list(np.geomspace(5e3, 1e7, 7))

    for dof in (5, 10, 15):
        S_0, S_1 = domain_for_dof(dof, alpha)
        print(f"\n{"=" * 70}\nRunning dof={dof}  (S_0={S_0:.2f}, S_1={S_1:.2f})\n{"=" * 70}")

        true_q = float(scipy_t.ppf(alpha, dof))
        est = dict(c_f=c_f, c_dg=c_dg, c_m=c_m, S_0=S_0, S_1=S_1, r=r, s_mh=s_mh, K=K)
        bl_arms = [(f"BL-HD (rho={rho:g})", studentt_sampler, (rho, dof), est)
                   for rho in sorted(rho_list)]
        baseline = (draw_studentt, (dof,), c_f)

        results = run_rmse_vs_cost(bl_arms, baseline, cost_grid, alpha, true_q,
                                   n_trials=n_trials, seed=0, verbose=True)

        tag = build_filename_tag(model="studentt", dof=dof, rho=rho_list, alpha=alpha,
                                 S0=round(S_0, 2), S1=round(S_1, 2), cf=c_f, cdg=c_dg, cm=c_m,
                                 smh=s_mh, ntrials=n_trials, K=K)
        # Cost-grid range appended so re-runs over a DIFFERENT grid don't overwrite each other.
        tag = f"{tag}_{cost_grid_tag(cost_grid)}"
        plot_labels = {f"BL-HD (rho={rho:g})": rf"BL-HD ($\varrho={rho:g}$)"
                       for rho in sorted(rho_list)}
        meta = dict(model="studentt", dof=dof, rho_list=rho_list, alpha=alpha, S_0=S_0, S_1=S_1,
                    r=r, c_f=c_f, c_dg=c_dg, c_m=c_m, s_mh=s_mh, n_trials=n_trials, K=K,
                    true_q=true_q, tag=tag, plot_labels=plot_labels)

        json_path, csv_path = save_results(results, meta, str(RESULTS_DIR), f"rmse_vs_cost_{tag}")
        print(f"Saved {json_path}\n      {csv_path}")

    print(f"\n{"=" * 70}\nAll three dof runs complete.\n{"=" * 70}")


if __name__ == "__main__":
    main()

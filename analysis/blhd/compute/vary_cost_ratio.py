"""Compute RMSE-vs-cost for BL-HD at a fixed correlation rho across several coarse-model cost ratios
c_dg (bivariate-normal model), against the HD/OS fine-only baselines. Writes results + meta to
results/blhd/ as JSON+CSV; plot with analysis/blhd/plot/rmse_vs_cost.py.

The c_dg dependence turns out to be a THRESHOLD/saturation effect, not a gradient: once the coarse
model is ~5x cheaper than the fine one, Var(q_HD) has stopped being the MSE bottleneck (the
correction term's estimation variance dominates), so making DG cheaper still buys essentially
nothing and the arms coincide on the O(C^-1/2) line. Only as c_f/c_dg -> 1 (coarse priced like a
full revaluation) does BL-HD visibly degrade -- and even then it still beats HD ~2x, because the
gain comes overwhelmingly from the fine/coarse CORRELATION, not from the coarse model's cheapness.
The portfolio's own ratio (c_f/c_dg = 210/1) sits deep in the saturated plateau.
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
    """Run the RMSE-vs-cost sweep across coarse-model cost ratios c_dg and save the results."""
    mu, sd, rho = 0.0, 1.0, 0.999
    alpha = 0.995
    S_0, S_1, r = 0.0, 4.0, 3
    c_f, c_m, s_mh = 250.0, 0.05, 0.05
    # c_dg values chosen so the interpretable ratio c_f/c_dg lands exactly on 1/5/50/250/1000
    # (250 matches the portfolio model's own c_f/c_dg = 210/1 most closely). The range spans the
    # saturation threshold: ratio=1 prices the coarse model exactly like a full revaluation (BL-HD's
    # premise gone -- the only arm that visibly degrades), while 5/50/250/1000 all sit in the
    # plateau where c_dg no longer matters. See the module docstring.
    cost_ratios = [1.0, 5.0, 50.0, 250.0, 1000.0]
    c_dg_list = [c_f / ratio for ratio in cost_ratios]
    n_trials, K = 50, 20
    cost_grid = list(np.geomspace(5e3, 3e7, 7))
    # The cheapest arm (c_dg=0.25) accumulates ~20M DG samples per trial at the top budget and peaks
    # around 2 GB RSS; the default worker count (= cpu_count) would multiply that into OOM territory.
    n_workers = 4

    true_q = float(norm.ppf(alpha, mu, sd))
    bl_arms = [(f"BL-HD (c_dg={cdg:g})", normal_sampler, (rho, mu, sd),
                dict(c_f=c_f, c_dg=cdg, c_m=c_m, S_0=S_0, S_1=S_1, r=r, s_mh=s_mh, K=K))
               for cdg in sorted(c_dg_list)]
    baseline = (draw_normal, (mu, sd), c_f)

    results = run_rmse_vs_cost(bl_arms, baseline, cost_grid, alpha, true_q,
                               n_trials=n_trials, seed=0, n_workers=n_workers, verbose=True)

    tag = build_filename_tag(model="normal", rho=rho, alpha=alpha, cf=c_f, cdg=c_dg_list,
                             cm=c_m, smh=s_mh, ntrials=n_trials, K=K)
    # Cost-grid range appended so re-runs over a DIFFERENT grid don't overwrite each other.
    tag = f"{tag}_{cost_grid_tag(cost_grid)}"
    # Legend shows the interpretable cost ratio c_f / c_c (how many coarse draws buy one fine
    # full-revaluation draw), rather than the raw coarse-model cost value. The coarse model is denoted
    # c (subscript 'c') throughout the plots, not 'dg'.
    plot_labels = {f"BL-HD (c_dg={cdg:g})": rf"BL-HD ($c_{{\mathrm{{f}}}}/c_{{\mathrm{{c}}}}={c_f / cdg:g}$)"
                   for cdg in sorted(c_dg_list)}
    meta = dict(model="normal", mu=mu, sd=sd, rho=rho, c_dg_list=c_dg_list, alpha=alpha,
                S_0=S_0, S_1=S_1, r=r, c_f=c_f, c_m=c_m, s_mh=s_mh, n_trials=n_trials, K=K,
                true_q=true_q, tag=tag, plot_labels=plot_labels)

    json_path, csv_path = save_results(results, meta, str(RESULTS_DIR), f"rmse_vs_cost_{tag}")
    print(f"\nSaved {json_path}\n      {csv_path}")


if __name__ == "__main__":
    main()

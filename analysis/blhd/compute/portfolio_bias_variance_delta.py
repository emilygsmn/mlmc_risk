"""Bias^2/variance decomposition of RMSE^2 vs. cost for BL-HD on the EIOPA-benchmark portfolio, using
the pure first-order DELTA approximation as the coarse model (gammas dropped) instead of delta-gamma.

This is the exact same analysis as portfolio_bias_variance.py -- identical cost frame, alphas,
n_trials, budget grid, cost parameters, and on-disk documentation (one JSON+CSV per alpha, same meta
schema, same 'bias_variance_' prefix so the existing plot scripts pick it up) -- with the SINGLE
difference that sampler["coarse"] is the delta model. The fine model (full revaluation) is unchanged,
so the pseudo-true VaR is identical and the cached accurate reference is reused as-is.

The cost frame is kept identical on purpose: c_dg=1 is reused as the coarse-model unit cost so the
delta and delta-gamma runs are compared at the very same budgets (a pure model swap, not a re-costing;
the delta model would in reality be a touch cheaper, but holding the frame fixed isolates the effect
of dropping the gamma term). Filenames get a 'delta' segment (bias_variance_delta_modelportfolio_...)
mirroring the delta-gamma run's 'deltagamma' segment (bias_variance_deltagamma_modelportfolio_...), so
the two coarse models are unambiguous on disk and the delta results stay out of the default
delta-gamma globs in the variance-reduction/efficiency plotters.

One JSON+CSV per alpha -> results/blhd/. Plot with analysis/blhd/plot/rmse_bias_std.py (RMSE/|bias|/std)
and analysis/blhd/plot/bias_variance.py (3-panel MSE decomposition). Requires network access. Sampler
+ domain built once, reused. RESUMABLE: an alpha whose output JSON already exists is skipped.
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import RESULTS_DIR

from mlmc_risk_estimation.blhd.samplers.portfolio import portfolio_sampler
from mlmc_risk_estimation.blhd.experiment import (run_bias_variance_vs_cost_portfolio,
                                                  calibrate_domain, estimate_pseudo_true_quantile,
                                                  save_results)
from mlmc_risk_estimation.blhd.plotting import build_filename_tag

ALPHAS = (0.9, 0.95, 0.99, 0.995, 0.999)


def _pseudo_true(sampler: dict, alpha: float, n_fallback: int, seed: int) -> float:
    """Accurate pseudo-true VaR from the cached large-pool reference (build_accurate_pseudotrue.py)
    if present; else a single noisy n_fallback-sample quantile. The fine model is identical to the
    delta-gamma run, so the same cached Q_full applies."""
    cache = RESULTS_DIR / "pseudotrue_accurate.json"
    if cache.exists():
        d = json.load(open(cache))
        if f"{alpha}" in d.get("Q", {}):
            print(f"  [pseudo-true] using cached accurate Q_full (N={d['n_total']:,}, "
                  f"SE~{d["Q_se"][f'{alpha}']:.4f})")
            return float(d["Q"][f"{alpha}"])
    print(f"  [pseudo-true] WARNING: no cache; falling back to noisy {n_fallback:,}-sample quantile. "
          "Run build_accurate_pseudotrue.py first.")
    return estimate_pseudo_true_quantile(sampler["fine"], alpha, n_fallback, seed=seed)


def main() -> None:
    """Run the bias/variance-vs-cost decomposition for BL-HD on the portfolio model with the delta
    coarse model, per alpha."""
    r = 3
    # Cost frame identical to the delta-gamma run (portfolio_bias_variance.py): c_dg is reused as the
    # coarse (delta) unit cost so both runs share the exact same budgets.
    c_f, c_dg, c_m, s_mh = 210.0, 1.0, 0.70, 40.0
    n_trials, K = 100, 20
    n_boot, n_pseudo_true, n_pilot, seed = 2000, 300_000, 200, 0
    cost_grid = list(np.geomspace(5e5, 1.5e7, 6))

    # Same model tag as the delta-gamma run ('portfolio'); the coarse model is distinguished by the
    # 'delta' filename segment (bias_variance_delta_...) added at save time below.
    def _tag(alpha: float) -> str:
        """Build the filename tag for this alpha's output files."""
        return build_filename_tag(model="portfolio", alpha=alpha, cf=c_f, cdg=c_dg, cm=c_m, smh=s_mh,
                                  ntrials=n_trials, K=K, npts=len(cost_grid),
                                  clo=int(round(cost_grid[0])), chi=int(round(cost_grid[-1])),
                                  nboot=n_boot)

    sampler = portfolio_sampler(coarse_model="delta")
    S_0, S_1, _ = calibrate_domain(sampler["fine"], n_pilot=n_pilot)
    print(f"[DELTA coarse model]  Domain [S_0,S_1]=[{S_0:.2f},{S_1:.2f}]  "
          f"cost_grid={[f'{c:,.0f}' for c in cost_grid]}\n")

    for alpha in ALPHAS:
        out_json = RESULTS_DIR / f"bias_variance_delta_{_tag(alpha)}.json"
        if out_json.exists():
            print(f"alpha={alpha}: already done ({out_json.name}) -- skipping (resume).\n")
            continue
        true_q = _pseudo_true(sampler, alpha, n_pseudo_true, seed)
        print(f"{"=" * 70}\nalpha={alpha}  pseudo-true VaR={true_q:.6f}\n{"=" * 70}")

        results = run_bias_variance_vs_cost_portfolio(
            sampler, alpha, cost_grid, S_0, S_1, r, c_f, c_dg, c_m, s_mh, true_q,
            n_trials=n_trials, K=K, seed=seed, verbose=True, n_boot=n_boot, with_baseline=True)

        meta = dict(model="portfolio_delta", coarse_model="delta", alpha=alpha, S_0=S_0, S_1=S_1,
                    r=r, c_f=c_f, c_dg=c_dg, c_m=c_m, s_mh=s_mh, n_trials=n_trials, K=K, n_boot=n_boot,
                    n_pseudo_true=n_pseudo_true, true_q=true_q, tag=_tag(alpha))

        json_path, csv_path = save_results(results, meta, str(RESULTS_DIR),
                                           f"bias_variance_delta_{_tag(alpha)}")
        print(f"Saved {json_path}\n      {csv_path}\n")

    print("All alpha runs complete.")


if __name__ == "__main__":
    main()

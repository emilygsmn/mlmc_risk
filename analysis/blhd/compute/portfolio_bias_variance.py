"""Bias^2/variance decomposition of RMSE^2 vs. cost for BL-HD on the EIOPA-benchmark portfolio,
across a WIDE cost range chosen to show the whole regime transition in one sweep: a low-cost end
where even a central confidence level (alpha=0.95) is still variance-limited (RMSE falling with
cost), and a high-cost end where even a far-tail level (alpha=0.995) starts to become bias-limited
(RMSE flattening onto its bias floor). Confirms that a flat RMSE-vs-cost curve is a bias floor
(bias^2 roughly constant while variance keeps shrinking) rather than a failure to converge.

bias/variance are computed from the per-trial errors e = q_hat - true_q: bias = mean(e) relative to
the pseudo-true VaR; variance = var(e) is the trial-to-trial scatter and does NOT depend on the
pseudo-true. BL-HD only (with_baseline=False): the HD/OS full-revaluation baselines are
prohibitively expensive at the high-cost end (N_fine ~ C_star / c_f full revals per trial) and are
not needed for the decomposition. One JSON+CSV per alpha is written to results/blhd/; plot the
units-consistent RMSE/|bias|/std view with analysis/blhd/plot/rmse_bias_std.py (and the 3-panel
MSE decomposition with analysis/blhd/plot/bias_variance.py). Requires network access. Sampler +
domain built once, reused.
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
    if present; else a single noisy n_fallback-sample quantile. A single-300k reference has SE ~0.2,
    the size of the RMSE floor, so measuring bias/RMSE against it inflates the apparent bias floor."""
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
    """Run the bias/variance-vs-cost decomposition for BL-HD on the portfolio model, per alpha."""
    r = 3
    c_f, c_dg, c_m, s_mh = 210.0, 1.0, 0.70, 40.0
    n_trials, K = 100, 20
    n_boot, n_pseudo_true, n_pilot, seed = 2000, 300_000, 200, 0
    # Top 1.5e7 (C/c_f ~ 71,400): reaches modestly higher than the 1e7 run while keeping the HD/OS
    # baselines (with_baseline=True -> N_fine = C/c_f full revals per trial) affordable. Memory is
    # flat (~3 GB, not leaking), so n_trials=100 is safe. Single-core; ~1-1.5h per alpha at 100
    # trials, ~5-6h for all four. RESUMABLE: an alpha whose output JSON already exists is skipped, so
    # a crash mid-run only loses the in-progress alpha and re-running finishes the rest. Low end 5e5
    # keeps the pilot MH chain N_m_p = C/(1e4 c_m) ~ 71 (>0). One run yields BOTH the bias/variance
    # decomposition AND the RMSE-vs-cost data (rmse per arm + HD/OS baselines).
    cost_grid = list(np.geomspace(5e5, 1.5e7, 6))

    def _tag(alpha: float) -> str:
        """Build the filename tag for this alpha's output files."""
        return build_filename_tag(model="portfolio", alpha=alpha, cf=c_f, cdg=c_dg, cm=c_m, smh=s_mh,
                                  ntrials=n_trials, K=K, npts=len(cost_grid),
                                  clo=int(round(cost_grid[0])), chi=int(round(cost_grid[-1])),
                                  nboot=n_boot)

    sampler = portfolio_sampler()
    S_0, S_1, _ = calibrate_domain(sampler["fine"], n_pilot=n_pilot)
    print(f"Domain [S_0,S_1]=[{S_0:.2f},{S_1:.2f}]  cost_grid={[f'{c:,.0f}' for c in cost_grid]}\n")

    # Coarse model is delta-gamma; the 'deltagamma' filename segment keeps these results distinct from
    # the delta-only run (portfolio_bias_variance_delta.py -> bias_variance_delta_...).
    for alpha in ALPHAS:
        out_json = RESULTS_DIR / f"bias_variance_deltagamma_{_tag(alpha)}.json"
        if out_json.exists():
            print(f"alpha={alpha}: already done ({out_json.name}) -- skipping (resume).\n")
            continue
        true_q = _pseudo_true(sampler, alpha, n_pseudo_true, seed)
        print(f"{"=" * 70}\nalpha={alpha}  pseudo-true VaR={true_q:.6f}\n{"=" * 70}")

        results = run_bias_variance_vs_cost_portfolio(
            sampler, alpha, cost_grid, S_0, S_1, r, c_f, c_dg, c_m, s_mh, true_q,
            n_trials=n_trials, K=K, seed=seed, verbose=True, n_boot=n_boot, with_baseline=True)

        meta = dict(model="portfolio", alpha=alpha, S_0=S_0, S_1=S_1, r=r, c_f=c_f, c_dg=c_dg,
                    c_m=c_m, s_mh=s_mh, n_trials=n_trials, K=K, n_boot=n_boot,
                    n_pseudo_true=n_pseudo_true, true_q=true_q, tag=_tag(alpha))

        json_path, csv_path = save_results(results, meta, str(RESULTS_DIR),
                                           f"bias_variance_deltagamma_{_tag(alpha)}")
        print(f"Saved {json_path}\n      {csv_path}\n")

    print("All alpha runs complete.")


if __name__ == "__main__":
    main()

"""Offline bias decomposition of the BL-HD estimator at a central quantile (alpha=0.95, where RMSE is
bias-limited). Splits E[q_hat] - Q_full into three telescoping terms via a substitution ladder
(one approximation swapped per rung, others held; population-scale reference objects remove MC noise):

  q_hat = q_HD + corr_op                              (actual)
  q_C   = q_HD    + corr_op                           = q_hat
  q_B   = q_OS    + corr_op        (OS base)          -> (c) HD-smoothing bias = q_C - q_B = q_HD - q_OS
  q_A   = q_OS    + corr_ref       (exact interps)    -> (b) smoothing+interp = q_B - q_A = corr_op - corr_ref
  q_0   = Q_full                                       -> (a) coupling/structural = q_A - Q_full
  total = q_C - Q_full = (a) + (b) + (c)

  corr_op  : the run's own correction (diagnostics).
  corr_ref : the correction recomputed with near-EXACT interpolants F_hat_ref (ECDF of a huge fine
             sample), rho_hat_ref (grid KDE), F_inv_dg_ref (empirical QF of a huge DG sample), and the
             run's own beta params a,b, via a long MH chain from pi_hat_ref.
  q_OS     : order-statistic quantile of the run's DG samples (diag['dg_q_os']).

Averaged over R_TRIALS seeds at the operating budget. Requires network access. Long job -- detached.
"""

import sys
import json
import warnings
from collections.abc import Callable
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from _common import RESULTS_DIR

from mlmc_risk_estimation.blhd.samplers.portfolio import portfolio_sampler
from mlmc_risk_estimation.blhd.experiment import calibrate_domain, estimate_pseudo_true_quantile
from mlmc_risk_estimation.blhd.baselines import order_statistic_quantile
from mlmc_risk_estimation.blhd.adaptive import adapt_tuning_blhd_quantile
from mlmc_risk_estimation.blhd.mh import build_log_pi_hat_from_callables, grw_metropolis_hastings

ALPHA = 0.95
C_BUDGET = 3e6            # bias-limited region (C/c_f ~ 14,000), where RMSE has hit its floor
R_TRIALS = 100           # (a)/(b) terms are high-variance; need many seeds to resolve (a) from 0
N_REF_FINE = 500_000      # huge fine sample for Q_full, F_hat_ref, rho_hat_ref
N_REF_DG = 1_000_000      # huge DG sample for Q_dg, F_inv_dg_ref
N_MH_REF = 60_000         # long MH chain for corr_ref
C_F, C_DG, C_M, S_MH, K, R = 210.0, 1.0, 0.70, 40.0, 20, 3
OUT = RESULTS_DIR / f"bias_decomposition_alpha{ALPHA}_C{int(C_BUDGET)}_ntrials{R_TRIALS}"


def _ecdf_callable(sorted_x: np.ndarray) -> Callable:
    """Build the empirical CDF of a sorted sample as an interpolating callable."""
    n = len(sorted_x)
    ranks = (np.arange(1, n + 1) - 0.5) / n
    return lambda x: np.interp(x, sorted_x, ranks, left=0.0, right=1.0)


def _emp_qf(sorted_x: np.ndarray) -> Callable:
    """Build the empirical quantile function of a sorted sample as an interpolating callable."""
    n = len(sorted_x)
    p = (np.arange(1, n + 1) - 0.5) / n
    return lambda u: np.interp(np.clip(np.asarray(u, float), 0.0, 1.0), p, sorted_x)


def _grid_kde(x: np.ndarray, S_0: float, S_1: float) -> Callable:
    """Near-exact density: gaussian KDE on a subsample, pre-evaluated on a grid and interpolated."""
    sub = np.random.default_rng(0).choice(x, size=min(len(x), 20000), replace=False)
    kde = gaussian_kde(sub)
    grid = np.linspace(S_0, S_1, 2000)
    vals = kde(grid)
    return lambda t: np.interp(t, grid, vals, left=0.0, right=0.0)


def main() -> None:
    """Run the bias decomposition over R_TRIALS seeds and write the running results to disk."""
    s = portfolio_sampler()
    S0, S1, _ = calibrate_domain(s["fine"], n_pilot=200)

    # reference (population) objects
    np.random.seed(1); fine = np.sort(s["fine"](N_REF_FINE))
    np.random.seed(2); dg = np.sort(s["coarse"](N_REF_DG))
    Q_full = order_statistic_quantile(fine, ALPHA)
    Q_dg = order_statistic_quantile(dg, ALPHA)
    F_hat_ref = _ecdf_callable(fine)
    rho_hat_ref = _grid_kde(fine, S0, S1)
    F_inv_dg_ref = _emp_qf(dg)
    print(f"Q_full={Q_full:.3f}  Q_dg={Q_dg:.3f}  (DG model bias {Q_dg-Q_full:+.3f})")

    def corr_ref(a: float, b: float, s_mh: float, start: float) -> float:
        """Recompute the BL-HD correction term with near-exact reference interpolants."""
        log_pi = build_log_pi_hat_from_callables(a, b, F_hat_ref, rho_hat_ref, S0, S1)
        chain, _ = grw_metropolis_hastings(log_pi, start, s_mh, N_MH_REF,
                                           burn_in=int(0.2 * N_MH_REF), rng=np.random.default_rng(0))
        return float(np.mean(chain - F_inv_dg_ref(F_hat_ref(chain))))

    a_terms, b_terms, c_terms, totals = [], [], [], []
    rng_master = np.random.default_rng(20260715)
    for t in range(R_TRIALS):
        seed = int(rng_master.integers(1 << 30))
        np.random.seed(seed)
        q_hat, d = adapt_tuning_blhd_quantile(
            ALPHA, C_BUDGET, s["coarse"], s["level0"], s["level1"], C_F, C_DG, C_M, S0, S1, R, S_MH,
            K=K, rng=np.random.default_rng(seed), verbose=False)
        q_HD, q_OS, corr_op = d["q_HD"], d["dg_q_os"], d["correction"]
        cref = corr_ref(d["a"], d["b"], d["s_mh_final"], q_hat)
        c_terms.append(q_HD - q_OS)              # (c) HD-smoothing bias
        b_terms.append(corr_op - cref)           # (b) smoothing+interpolation bias
        a_terms.append(q_OS + cref - Q_full)     # (a) coupling/structural bias
        totals.append(q_hat - Q_full)            # total bias

        # Crash-safe: dump raw per-trial terms + running means every trial, so a mid-run crash
        # (the machine has been dropping) preserves all completed seeds.
        def _ms(v: list[float]) -> tuple[float, float]:
            """Return the (mean, standard-error-of-mean) of a list of trial values."""
            v = np.asarray(v)
            return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 1 else (float(v[0]), float("nan"))
        (am, ae), (bm, be), (cm, ce), (tm, te) = map(_ms, (a_terms, b_terms, c_terms, totals))
        json.dump(dict(alpha=ALPHA, C=C_BUDGET, r_trials_done=t + 1, r_trials_target=R_TRIALS,
                       Q_full=Q_full, Q_dg=Q_dg,
                       a=am, a_se=ae, b=bm, b_se=be, c=cm, c_se=ce, total=tm, total_se=te,
                       a_terms=a_terms, b_terms=b_terms, c_terms=c_terms, totals=totals),
                  open(str(OUT) + ".json", "w"), indent=2)
        if (t + 1) % 5 == 0:
            print(f"  {t+1}/{R_TRIALS} done   (a={am:+.4f}+/-{ae:.4f}  b={bm:+.4f}+/-{be:.4f}  "
                  f"c={cm:+.4f})", flush=True)

    def ms(v: list[float]) -> tuple[float, float]:
        """Return the (mean, standard-error-of-mean) of a list of trial values."""
        v = np.asarray(v); return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
    (a_m, a_e), (b_m, b_e), (c_m, c_e), (tot_m, tot_e) = map(ms, (a_terms, b_terms, c_terms, totals))
    print(f"\nBias decomposition at alpha={ALPHA}, C={C_BUDGET:.0e} ({R_TRIALS} seeds):")
    print(f"  (a) coupling/structural : {a_m:+.4f} +/- {a_e:.4f}")
    print(f"  (b) smoothing+interp    : {b_m:+.4f} +/- {b_e:.4f}")
    print(f"  (c) HD-smoothing        : {c_m:+.4f} +/- {c_e:.4f}")
    print(f"  sum (a)+(b)+(c)         : {a_m+b_m+c_m:+.4f}")
    print(f"  total  E[q_hat]-Q_full  : {tot_m:+.4f} +/- {tot_e:.4f}   (closure check)")

    json.dump(dict(alpha=ALPHA, C=C_BUDGET, r_trials=R_TRIALS, Q_full=Q_full, Q_dg=Q_dg,
                   a=a_m, a_se=a_e, b=b_m, b_se=b_e, c=c_m, c_se=c_e, total=tot_m, total_se=tot_e),
              open(str(OUT) + ".json", "w"), indent=2)
    print("Saved", str(OUT) + ".json")


if __name__ == "__main__":
    main()
    print("ALL DONE")

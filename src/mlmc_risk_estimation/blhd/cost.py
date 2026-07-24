"""Benchmark measuring the three per-unit wall-clock costs the BL-HD RMSE comparison needs,
all normalized to c_dg = 1:

  c_dg : cost of one Delta-Gamma revaluation (the unit).
  c_f  : cost of one full revaluation, in c_dg units.
  c_m  : cost of one Metropolis-Hastings step, in c_dg units.

Method:
  - Greeks are computed once (they do not depend on the MC scenarios)
  - Full vs. DG revaluation are timed on the same pre-generated scenario batch,
    so the shared scenario-generation cost is excluded from both. We compare
    pure "revaluation given shocks" cost, which is what c_f / c_dg represent.
  - One MH step is timed by running grw_metropolis_hastings for many steps on a
    representative log_pi_hat (built from a pilot fine batch via the real
    build_pi_hat) and dividing by the step count.
  - Every cost is timed over repeated runs after a warm-up. The minimum per-unit
    time is the point estimate (least noisy). The coefficient of variation across
    repeats is reported as an explicit stability check.

Call measure_costs() to obtain a dict with c_f, c_dg (=1), c_m and the raw timings.
"""

import time
from collections.abc import Callable

import numpy as np

from mlmc_risk_estimation.portfolio.utils.io_helpers import read_config, get_portfolio, get_instr_info
from mlmc_risk_estimation.portfolio.utils.preproc_helpers import preproc_portfolio, get_historical_data
from mlmc_risk_estimation.portfolio.model_calibration import calibrate_models
from mlmc_risk_estimation.portfolio.scenario_generation import generate_mc_shocks_pycopula
from mlmc_risk_estimation.portfolio.full_valuation import calc_prices
from mlmc_risk_estimation.portfolio.deltagamma_valuation import compute_greeks, apply_delta_gamma_pnl
from mlmc_risk_estimation.portfolio.risk_aggregation import calc_instr_pnls, calc_portfolio_pnl

from mlmc_risk_estimation.blhd.mh import build_log_pi_hat_from_callables, grw_metropolis_hastings
from mlmc_risk_estimation.blhd.gnr import build_monotone_cdf_interp, build_nonneg_pdf_interp

__all__ = ["measure_costs"]


def _time_per_unit(fn: Callable[[], object], n_units: int, repeats: int) -> np.ndarray:
    """Runs fn() `repeats` times (after one warm-up) and returns the array of
    per-unit wall-clock times in seconds (total time / n_units per run)."""
    fn()  # Run the untimed warm-up
    times = np.empty(repeats)
    for i in range(repeats):
        t0 = time.perf_counter()
        fn()
        times[i] = (time.perf_counter() - t0) / n_units
    return times


def _build_representative_log_pi_hat(fine_pilot: np.ndarray, S_0: float, S_1: float, alpha: float,
                                      r: int = 3) -> object:
    """Builds the MH target density the production estimator actually samples: GNR's CDF
    interpolant (build_monotone_cdf_interp) and density interpolant (build_nonneg_pdf_interp)
    on equidistant knot grids of GNR's k = deg*n+1 shape.
    """
    r_cdf = r + 1
    x_sorted = np.sort(fine_pilot)
    n = x_sorted.size

    s_cdf = np.linspace(S_0, S_1, r_cdf * 40 + 1)
    F_vals = np.searchsorted(x_sorted, s_cdf, side="right") / n

    counts, edges = np.histogram(fine_pilot, bins=400, range=(S_0, S_1), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    s_pdf = np.linspace(S_0, S_1, r * 40 + 1)
    rho_vals = np.interp(s_pdf, centers, counts)

    # HD Beta kernel at the tail level p=1-alpha for a representative sample count.
    p = 1 - alpha
    n_kernel = 1000
    a, b = p * (n_kernel + 1), (1 - p) * (n_kernel + 1)

    F_hat = build_monotone_cdf_interp(s_cdf, F_vals, r_cdf)
    rho_hat = build_nonneg_pdf_interp(s_pdf, rho_vals, r)
    return build_log_pi_hat_from_callables(a, b, F_hat, rho_hat, S_0, S_1)


def measure_costs(n_scen: int = 20_000, n_mh_steps: int = 200_000, repeats: int = 7,
              alpha: float = 0.995, s_mh: float = 30.0, domain_margin: float = 1.5) -> dict:
    """Measure the three per-unit costs (c_dg, c_f, c_m) the BL-HD RMSE comparison needs and print
    the timings.
    """

    path_config = read_config("data/config/path.yaml")
    param_config = read_config(path_config["input"]["param_config"])
    param_config["monte_carlo"]["n"] = n_scen

    portfolio = get_portfolio(path_config["input"], param_config)
    instr_info = get_instr_info(path_config["input"])
    portfolio, instr_info, der_underlyings, weights = preproc_portfolio(portfolio=portfolio,
                                                                        instr_info=instr_info)
    hist_data = get_historical_data(path_config, param_config, instr_info)
    instr_info, calib_param = calibrate_models(hist_data, instr_info, param_config)
    val_date = param_config["valuation"]["val_date"]

    base_values = calc_prices(mkt_data=hist_data, instr_info=instr_info, ref_date=val_date,
                              param_config=param_config, der_underlyings=der_underlyings,
                              shocks=None)

    # One shared scenario batch, reused for both reval models
    mc_scenarios = generate_mc_shocks_pycopula(hist_data, instr_info, param_config, calib_param,
                                               ref_date=val_date)

    # Compute the Greeks once
    t0 = time.perf_counter()
    deltas, gammas = compute_greeks(hist_data, instr_info, val_date, param_config,
                                    der_underlyings, weights)
    t_greeks_fixed = time.perf_counter() - t0

    def full_reval() -> object:
        """Full revaluation of the portfolio PnL on the shared scenario batch."""
        shocked = calc_prices(mkt_data=hist_data, instr_info=instr_info, ref_date=val_date,
                              param_config=param_config, der_underlyings=der_underlyings,
                              shocks=mc_scenarios)
        instr_pnls = calc_instr_pnls(prices_at_t1=base_values, prices_at_t2=shocked)
        return calc_portfolio_pnl(instr_pnls=instr_pnls, weights=weights)

    def dg_reval() -> np.ndarray:
        """Delta-gamma revaluation of the portfolio PnL on the shared scenario batch."""
        return apply_delta_gamma_pnl(deltas, gammas, mc_scenarios)

    t_full = _time_per_unit(full_reval, n_scen, repeats)
    t_dg = _time_per_unit(dg_reval, n_scen, repeats)

    # Metropolis-Hastings step cost
    fine_pilot = full_reval().to_numpy().ravel()
    lo, hi = fine_pilot.min(), fine_pilot.max()
    span = hi - lo
    S_0, S_1 = lo - domain_margin * span, hi + domain_margin * span
    log_pi_hat = _build_representative_log_pi_hat(fine_pilot, S_0, S_1, alpha)

    x1 = float(np.quantile(fine_pilot, 1 - alpha))
    if not np.isfinite(log_pi_hat(x1)):
        x1 = float(np.median(fine_pilot))

    def mh_run() -> tuple[np.ndarray, float]:
        """Run n_mh_steps of the GRW Metropolis-Hastings sampler with no burn-in (clean per-step
        timing)."""
        return grw_metropolis_hastings(log_pi_hat, x1, s_mh, n_mh_steps,
                                       burn_in=0, rng=np.random.default_rng())

    t_mh = _time_per_unit(mh_run, n_mh_steps, repeats)

    # Uncompiled GNR path (Python callable loop)
    t_mh_py = None
    gnr_data = log_pi_hat.__dict__.pop("_njit_gnr_data", None)
    if gnr_data is not None:
        n_py = max(n_mh_steps // 20, 5000)

        def mh_run_py() -> tuple[np.ndarray, float]:
            """Run n_py steps of the uncompiled (Python-callable) GRW Metropolis-Hastings sampler."""
            return grw_metropolis_hastings(log_pi_hat, x1, s_mh, n_py,
                                           burn_in=0, rng=np.random.default_rng())

        t_mh_py = _time_per_unit(mh_run_py, n_py, max(repeats // 2, 3))
        # Restore the compiled path
        log_pi_hat._njit_gnr_data = gnr_data

    # Point estimates (minimum) and normalization.
    t_full_min, t_dg_min, t_mh_min = t_full.min(), t_dg.min(), t_mh.min()
    c_dg = 1.0
    c_f = t_full_min / t_dg_min
    c_m = t_mh_min / t_dg_min

    def cv(x: np.ndarray) -> float:
        """Coefficient of variation (std/mean) of an array of timings."""
        return x.std() / x.mean()

    print(f"Portfolio: {instr_info.shape[0]} instruments, {mc_scenarios.shape[1]} risk factors")
    print(f"Reval batch: {n_scen:,} scenarios | MH: {n_mh_steps:,} steps | "
          f"{repeats} repeats")
    print("MH density: GNR interpolants (build_monotone_cdf_interp / build_nonneg_pdf_interp)\n")
    print(f"{"unit":<28}{"min time":>14}{"CV (std/mean)":>16}")
    print(f"{"full revaluation":<28}{t_full_min * 1e6:>11.3f} us{cv(t_full):>15.1%}")
    print(f"{"delta-gamma reval":<28}{t_dg_min * 1e6:>11.3f} us{cv(t_dg):>15.1%}")
    print(f"{"MH step (compiled GNR)":<28}{t_mh_min * 1e6:>11.3f} us{cv(t_mh):>15.1%}")
    if t_mh_py is not None:
        print(f"{"MH step (uncompiled GNR)":<28}{t_mh_py.min() * 1e6:>11.3f} us{cv(t_mh_py):>15.1%}"
              f"   <- Python-callable path (before)")
    print()
    print(f"  c_dg = {c_dg:.1f}  (unit)")
    print(f"  c_f  = {c_f:8.2f}  (one full reval = {c_f:.0f} DG revals)")
    print(f"  c_m  = {c_m:8.4f}  (one MH step = {c_m:.3f} DG revals)")
    if t_mh_py is not None:
        print(f"  c_m was {t_mh_py.min() / t_dg_min:8.4f} uncompiled "
              f"-> {t_mh_py.min() / t_mh_min:.0f}x faster compiled")
    print()
    print(f"c_m stability: CV = {cv(t_mh):.1%} across {repeats} repeats "
          f"-> {"stable" if cv(t_mh) < 0.15 else "noisy, increase n_mh_steps/repeats"} "
          f"(direct per-step timing, no differencing)")
    
    print(f"\nOne-time Greek build (fixed overhead): "
          f"{t_greeks_fixed * 1e3:.1f} ms = {t_greeks_fixed / t_dg_min:,.0f} DG scenarios' worth")

    return dict(c_f=c_f, c_dg=c_dg, c_m=c_m,
                t_full=t_full, t_dg=t_dg, t_mh=t_mh, t_mh_py=t_mh_py, t_greeks=t_greeks_fixed)

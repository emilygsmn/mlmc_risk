"""Diagnostics comparing the portfolio's full-valuation loss L^f against its Delta (L^d) and
Delta-Gamma (L^dg) approximations, all on ONE shared Monte Carlo scenario draw (like-for-like).
Plots the loss L = -PnL (positive = loss), matching the thesis convention. Produces four figures in
the project house style (no title, project colours, serif/CM math via apply_style):

  1. Q-Q plot          full-val. loss quantiles vs Delta (grey) and Delta-Gamma (black), y=x in red.
  2. Histogram         overlaid unfilled step outlines: Delta grey, Delta-Gamma black, full red.
  3. Scatter (Delta)       raw unsorted samples: full-valuation loss L^f (x) vs Delta loss L^d (y).
  4. Scatter (Delta-Gamma) raw unsorted samples: full-valuation loss L^f (x) vs Delta-Gamma loss L^dg (y).

Colour convention throughout: Delta = grey (0.45), Delta-Gamma = black, full valuation / y=x = red.
Requires network access. PNGs are written to results/blhd/.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import RESULTS_DIR

from mlmc_risk_estimation.portfolio.utils.io_helpers import read_config, get_portfolio, get_instr_info
from mlmc_risk_estimation.portfolio.utils.preproc_helpers import preproc_portfolio, get_historical_data
from mlmc_risk_estimation.portfolio.model_calibration import calibrate_models
from mlmc_risk_estimation.portfolio.scenario_generation import generate_mc_shocks_pycopula
from mlmc_risk_estimation.portfolio.full_valuation import calc_prices
from mlmc_risk_estimation.portfolio.deltagamma_valuation import (calc_delta_scenario_pnl,
                                                                 calc_delta_gamma_scenario_pnl)
from mlmc_risk_estimation.portfolio.risk_aggregation import calc_instr_pnls, calc_portfolio_pnl

from mlmc_risk_estimation.blhd.plotting import apply_style, SAVE_DPI, build_filename_tag

N_SCEN = 50000
DELTA_COLOR = "0.45"
DG_COLOR = "black"
FULL_COLOR = "red"

# Identical axes rectangle for every figure so their frames match exactly. tight_layout() picks
# per-plot margins (varying with tick-label widths) and set_aspect(..., adjustable="box") resizes
# the axes box, so we avoid both and pin the rectangle by hand. Equal width/height fractions on the
# square 6x6 figure give a square box -> the y=x diagonal stays at 45 deg without set_aspect.
# Left margin increased from 0.15 to 0.22 to accommodate larger y-axis labels from apply_style().
_MARGINS = dict(left=0.22, right=0.97, bottom=0.13, top=0.95)


def get_full_delta_and_delta_gamma_pnl(
        n_scen: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Run the portfolio model and return (full_pnl, delta_pnl, dg_pnl, val_date), all on the SAME
    Monte Carlo scenarios so the three P&L series are directly comparable scenario-by-scenario."""
    path_config = read_config("data/config/path.yaml")
    param_config = read_config("data/config/param.yaml")
    param_config["monte_carlo"]["n"] = n_scen

    portfolio = get_portfolio(path_config["input"], param_config)
    instr_info = get_instr_info(path_config["input"])
    portfolio, instr_info, der_underlyings, weights = preproc_portfolio(portfolio=portfolio,
                                                                        instr_info=instr_info)
    hist_data = get_historical_data(path_config, param_config, instr_info)
    instr_info, calib_param = calibrate_models(hist_data, instr_info, param_config)

    val_date = param_config["valuation"]["val_date"]
    mc_scenarios = generate_mc_shocks_pycopula(hist_data, instr_info, param_config, calib_param,
                                               ref_date=val_date)

    base_values = calc_prices(mkt_data=hist_data, instr_info=instr_info, ref_date=val_date,
                              param_config=param_config, der_underlyings=der_underlyings, shocks=None)
    shocked_values = calc_prices(mkt_data=hist_data, instr_info=instr_info, ref_date=val_date,
                                 param_config=param_config, der_underlyings=der_underlyings,
                                 shocks=mc_scenarios)
    instr_pnls = calc_instr_pnls(prices_at_t1=base_values, prices_at_t2=shocked_values)
    full_pnl = calc_portfolio_pnl(instr_pnls=instr_pnls, weights=weights)["total_pnl"].to_numpy()

    delta_pnl = calc_delta_scenario_pnl(mkt_data=hist_data, instr_info=instr_info, ref_date=val_date,
                                        param_config=param_config, der_underlyings=der_underlyings,
                                        scenario_shocks=mc_scenarios, weights=weights)["pnl"].to_numpy()
    dg_pnl = calc_delta_gamma_scenario_pnl(mkt_data=hist_data, instr_info=instr_info,
                                           ref_date=val_date, param_config=param_config,
                                           der_underlyings=der_underlyings,
                                           scenario_shocks=mc_scenarios, weights=weights)["pnl"].to_numpy()
    return full_pnl, delta_pnl, dg_pnl, val_date


def _save(fig: plt.Figure, name: str, n_scen: int, val_date: object) -> None:
    """Save a figure to results/blhd/{name}_{tag}.png, tagged by n_scen and val_date."""
    tag = build_filename_tag(n=n_scen, val_date=val_date)
    path = str(RESULTS_DIR / f"{name}_{tag}.png")
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"Saved {path}")


def plot_qq(full_pnl: np.ndarray, delta_pnl: np.ndarray, dg_pnl: np.ndarray, n_scen: int,
           val_date: object) -> None:
    """Q-Q plot: sorted full-valuation quantiles vs the Delta and Delta-gamma quantiles."""
    full_s, delta_s, dg_s = np.sort(full_pnl), np.sort(delta_pnl), np.sort(dg_pnl)
    lo = min(full_s.min(), delta_s.min(), dg_s.min()) - 10
    hi = max(full_s.max(), delta_s.max(), dg_s.max()) + 10

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(full_s, delta_s, linestyle="none", marker="o", markersize=4, markerfacecolor="none",
            color=DELTA_COLOR, markeredgecolor=DELTA_COLOR, label=r"$L^{\mathrm{d}}$", zorder=1)
    ax.plot(full_s, dg_s, linestyle="none", marker="o", markersize=4, markerfacecolor="none",
            color=DG_COLOR, label=r"$L^{\mathrm{dg}}$", zorder=2)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=FULL_COLOR, label="$y=x$", zorder=3)

    ax.set_xlabel(r"$L^{\mathrm{f}}_{i:n}$")
    ax.set_ylabel(r"$L^{\mathrm{c}}_{i:n}$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend()
    fig.subplots_adjust(**_MARGINS)
    _save(fig, "qq_full_vs_delta_and_delta_gamma", n_scen, val_date)


def plot_hist(full_pnl: np.ndarray, delta_pnl: np.ndarray, dg_pnl: np.ndarray, n_scen: int,
             val_date: object, n_bins: int = 120) -> None:
    """Overlaid, unfilled (step-outline) histograms of the three P&L series on shared bin edges."""
    lo = min(full_pnl.min(), delta_pnl.min(), dg_pnl.min()) - 10
    hi = max(full_pnl.max(), delta_pnl.max(), dg_pnl.max()) + 10
    bins = np.linspace(lo, hi, n_bins + 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(delta_pnl, bins=bins, density=True, histtype="step", color=DELTA_COLOR,
            label=r"$L^{\mathrm{d}}$")
    ax.hist(dg_pnl, bins=bins, density=True, histtype="step", color=DG_COLOR,
            label=r"$L^{\mathrm{dg}}$")
    ax.hist(full_pnl, bins=bins, density=True, histtype="step", color=FULL_COLOR,
            label=r"$L^{\mathrm{f}}$")

    ax.set_xlabel(r"$L$")
    ax.set_ylabel("Density")
    ax.legend()
    fig.subplots_adjust(**_MARGINS)
    _save(fig, "hist_full_delta_delta_gamma", n_scen, val_date)


def plot_scatter(full_pnl: np.ndarray, approx_pnl: np.ndarray, name: str, color: str, sym: str,
                 n_scen: int, val_date: object) -> None:
    """Per-scenario scatter of the raw, UNSORTED samples: full-valuation loss L^f (x) vs the
    approximation's loss (y). Each point is one scenario, so an off-diagonal point is that scenario's
    approximation error -- unlike the Q-Q plot, which sorts each series independently and so hides
    per-scenario mismatches. The red y=x line marks exact agreement. `sym` is the loss's math symbol
    (e.g. r"L^{\\mathrm{d}}")."""
    lo = min(full_pnl.min(), approx_pnl.min()) - 10
    hi = max(full_pnl.max(), approx_pnl.max()) + 10

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(full_pnl, approx_pnl, linestyle="none", marker="o", markersize=3, markerfacecolor="none",
            color=color, markeredgecolor=color, label=rf"${sym}$", zorder=1)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=FULL_COLOR, label="$y=x$", zorder=2)

    ax.set_xlabel(r"$L^{\mathrm{f}}$")
    ax.set_ylabel(rf"${sym}$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend()
    fig.subplots_adjust(**_MARGINS)
    _save(fig, name, n_scen, val_date)


def main() -> None:
    """Run the shared-scenario PnL comparison and produce the Q-Q, histogram, and scatter figures."""
    apply_style()
    full_pnl, delta_pnl, dg_pnl, val_date = get_full_delta_and_delta_gamma_pnl(N_SCEN)
    # Work with the loss L = -PnL (positive = loss), matching the thesis convention where VaR is the
    # upper-tail quantile of the loss L^f / L^d / L^dg.
    full_loss, delta_loss, dg_loss = -full_pnl, -delta_pnl, -dg_pnl
    plot_qq(full_loss, delta_loss, dg_loss, N_SCEN, val_date)
    plot_hist(full_loss, delta_loss, dg_loss, N_SCEN, val_date)
    # Standalone delta scatter: black (grey only distinguishes delta from delta-gamma when both
    # appear together, as in the Q-Q plot and histogram).
    plot_scatter(full_loss, delta_loss, "scatter_full_vs_delta", DG_COLOR,
                 r"L^{\mathrm{d}}", N_SCEN, val_date)
    plot_scatter(full_loss, dg_loss, "scatter_full_vs_delta_gamma", DG_COLOR,
                 r"L^{\mathrm{dg}}", N_SCEN, val_date)


if __name__ == "__main__":
    main()

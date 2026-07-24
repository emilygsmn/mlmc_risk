"""Plot the final CDF and PDF that the adaptive BL-HD procedure ends up using: the GNR MLMC
interpolants F_hat and rho_hat of the portfolio loss L^f over the estimation domain [S_0, S_1].
Runs the estimator once on the portfolio (loss convention) at a fixed budget, pulls F_hat / rho_hat
from the returned diagnostics, and plots them (with the estimated VaR marked). House style, high-res.
Two PNGs -> results/blhd/. Requires network access.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import RESULTS_DIR

from mlmc_risk_estimation.blhd.samplers.portfolio import portfolio_sampler
from mlmc_risk_estimation.blhd.experiment import calibrate_domain
from mlmc_risk_estimation.blhd.adaptive import adapt_tuning_blhd_quantile
from mlmc_risk_estimation.blhd.plotting import apply_style, SAVE_DPI, build_filename_tag, set_frame_height

ALPHA = 0.995
BUDGETS = (1e4, 1e7)
C_F, C_DG, C_M, S_MH, K, R = 210.0, 1.0, 0.70, 40.0, 20, 3


def _save(fig: matplotlib.figure.Figure, name: str, tag: str) -> None:
    """Save a figure to results/blhd/ under a name_tag.png filename and close it."""
    path = str(RESULTS_DIR / f"{name}_{tag}.png")
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_for_budget(sampler: dict, S_0: float, S_1: float, C_star: float) -> None:
    """Run BL-HD once at the given budget and save its CDF and PDF plots."""
    np.random.seed(0)
    q_hat, diag = adapt_tuning_blhd_quantile(
        ALPHA, C_star, sampler["coarse"], sampler["level0"], sampler["level1"],
        C_F, C_DG, C_M, S_0, S_1, R, S_MH, K=K, rng=np.random.default_rng(0), verbose=False)

    F_hat, rho_hat, s0, s1 = diag["F_hat"], diag["rho_hat"], diag["S_0"], diag["S_1"]
    grid = np.linspace(s0, s1, 2000)
    print(f"C={C_star:.0e}  VaR q_hat={q_hat:.3f}  F_hat(q_hat)={float(F_hat(q_hat)):.4f}")

    tag = build_filename_tag(model="portfolio", alpha=ALPHA, C=int(C_star))

    # CDF
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(grid, F_hat(grid), color="black", lw=1.8, label=r"$\widehat{F}_{L^{\mathrm{f}}}$")
    ax.axhline(ALPHA, color="0.6", linestyle=":", lw=1.0)
    ax.axvline(q_hat, color="black", linestyle="--", lw=1.0,
               label=r"$\widehat{\mathrm{VaR}}_{\alpha}$")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\widehat{F}_{L^{\mathrm{f}}}$")
    ax.set_xlim(s0, s1)
    ax.legend()
    set_frame_height(fig)
    _save(fig, "cdf_estimate", tag)

    # PDF
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(grid, rho_hat(grid), color="black", lw=1.8, label=r"$\widehat{f}_{L^{\mathrm{f}}}$")
    ax.axvline(q_hat, color="black", linestyle="--", lw=1.0,
               label=r"$\widehat{\mathrm{VaR}}_{\alpha}$")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\widehat{f}_{L^{\mathrm{f}}}$")
    ax.set_xlim(s0, s1)
    ax.legend()
    set_frame_height(fig)
    _save(fig, "pdf_estimate", tag)


def main() -> None:
    """Plot the BL-HD CDF/PDF estimates at each budget in BUDGETS."""
    apply_style()
    sampler = portfolio_sampler()
    S_0, S_1, _ = calibrate_domain(sampler["fine"], n_pilot=200)
    for C_star in BUDGETS:
        _plot_for_budget(sampler, S_0, S_1, C_star)


if __name__ == "__main__":
    main()

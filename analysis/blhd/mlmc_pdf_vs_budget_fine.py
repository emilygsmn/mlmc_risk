"""MLMC PDF rho_hat the BL-HD estimator builds, overlaid across a FINER budget ladder
(5e5, 1e6, 5e6, 1e7) than mlmc_density_plots.py -- chosen to sit in the range where the PDF
interpolant actually starts to refine (below ~5e5 the coarse budgets all yield the identical PDF).
Portfolio, loss convention, alpha=0.995. All curves black, distinguished by line style. House style,
(6,6), high-res. One PNG -> results/blhd/mlmc_pdf_vs_budget_fine.png. Requires network access.
"""

import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import RESULTS_DIR

from mlmc_risk_estimation.blhd.samplers.portfolio import portfolio_sampler
from mlmc_risk_estimation.blhd.experiment import calibrate_domain
from mlmc_risk_estimation.blhd.adaptive import adapt_tuning_blhd_quantile
from mlmc_risk_estimation.blhd.plotting import apply_style, SAVE_DPI, set_frame_height

ALPHA = 0.995
BUDGETS = (5e5, 1e6, 5e6, 1e7)
# All budget curves are black; distinguished by line style.
LINESTYLES = {5e5: ":", 1e6: "-.", 5e6: "--", 1e7: "-"}
C_F, C_DG, C_M, S_MH, K, R = 210.0, 1.0, 0.70, 40.0, 20, 3


def _run(sampler: dict[str, Any], S_0: float, S_1: float, C_star: float) -> dict[str, Any]:
    """Run the BL-HD quantile estimator once at budget C_star and return its diagnostics dict."""
    np.random.seed(0)
    _, diag = adapt_tuning_blhd_quantile(
        ALPHA, C_star, sampler["coarse"], sampler["level0"], sampler["level1"],
        C_F, C_DG, C_M, S_0, S_1, R, S_MH, K=K, rng=np.random.default_rng(0), verbose=False)
    return diag


def _budget_label(C: float) -> str:
    """LaTeX label like $5 \\times 10^{5}$ / $10^{6}$ for a general (not necessarily power-of-ten) C."""
    exp = int(np.floor(np.log10(C)))
    mant = C / 10 ** exp
    if abs(mant - 1.0) < 1e-9:
        return rf"$\mathcal{{C}}_{{\mathrm{{total}}}}=10^{{{exp}}}$"
    return rf"$\mathcal{{C}}_{{\mathrm{{total}}}}={mant:g}\times 10^{{{exp}}}$"


def main() -> None:
    """Run the estimator across the fine budget ladder and save the overlaid PDF figure."""
    apply_style()
    sampler = portfolio_sampler()
    S_0, S_1, _ = calibrate_domain(sampler["fine"], n_pilot=200)
    grid = np.linspace(S_0, S_1, 2000)

    diags = {}
    for C in BUDGETS:
        diags[C] = _run(sampler, S_0, S_1, C)
        print(f"C={C:.1e}  VaR={diags[C]['q_hat']:.2f}  peak={diags[C]['rho_hat'](grid).max():.6f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    # Draw fine -> coarse so broader low-budget curves stay visible where converged high-budget
    # curves would otherwise cover them.
    for C in sorted(BUDGETS, reverse=True):
        ax.plot(grid, diags[C]["rho_hat"](grid), color="black", lw=1.5, linestyle=LINESTYLES[C],
                label=_budget_label(C))
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\widehat{f}_{L^{\mathrm{f}}}$")
    ax.set_xlim(S_0, S_1)
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], lbls[::-1])   # ascending budget in the legend
    set_frame_height(fig)
    path = str(RESULTS_DIR / "mlmc_pdf_vs_budget_fine.png")
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

"""MLMC level-1 variance profile V_1(x) of the coupled fine/coarse difference, vs loss level x --
the quantity that governs how many expensive full-valuation samples GNR needs at each evaluation
point (small V_1 -> GNR works better). Uses ONE shared 50k-scenario draw of coupled (L^f, L^dg)
pairs (loss convention), and one BL-HD run to read the smoothing width delta / knot count / degree r
that the adaptive algorithm actually converges to at a representative budget.

Two figures (house style, high-res), each V_1(x) vs x over [S_0, S_1]:
  1. CDF: V_1(x) = Var[ g(L^f) - g(L^dg) ] with g the indicator 1_{L<=x} (unsmoothed) and with GNR's
     mollified step g_cdf((L-x)/delta; r) (smoothed, as used). Indicator case = p(x)(1-p(x)).
  2. PDF: V_1(x) with g the raw boxcar bin 1_{|L-x|<delta}/(2delta) (unsmoothed density) and GNR's
     mollified Dirac g_pdf((L-x)/delta; r)/delta (smoothed, as used).

Requires network access. PNGs -> results/blhd/.
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
from pnl_approximation_diagnostics import get_full_delta_and_delta_gamma_pnl

from mlmc_risk_estimation.blhd.samplers.portfolio import portfolio_sampler
from mlmc_risk_estimation.blhd.experiment import calibrate_domain
from mlmc_risk_estimation.blhd.adaptive import adapt_tuning_blhd_quantile
from mlmc_risk_estimation.blhd.mlmc_cdf import _g_for_cdf
from mlmc_risk_estimation.blhd.mlmc_pdf import _g_for_pdf
from mlmc_risk_estimation.blhd.plotting import apply_style, SAVE_DPI, set_frame_height, LBL_VAR

ALPHA = 0.995
C_BUDGET = 1e6          # budget whose converged smoothing width delta the plot uses
N_SCEN = 50_000
N_GRID = 500
ALPHAS = (0.9, 0.95, 0.99, 0.995)   # quantile levels marked on the u-axis plots
U_LO = 0.8              # lower u shown on the quantile-level (tail) plots
C_F, C_DG, C_M, S_MH, K, R = 210.0, 1.0, 0.70, 40.0, 20, 3

# Legend labels (grey = unsmoothed functional, black = GNR-smoothed), shared by both x-axes.
LBL_CDF = (r"$\mathbf{1}_{\{L^{\mathrm{f}}\leq s\}} - \mathbf{1}_{\{L^{\mathrm{dg}}\leq s\}}$",
           r"$\psi\,\left(\frac{L^{\mathrm{f}}-s}{\gamma}\right) - \psi\,\left(\frac{L^{\mathrm{dg}}-s}{\gamma}\right)$")
LBL_PDF = (r"$\frac{1}{2\gamma}\left(\mathbf{1}_{\{|L^{\mathrm{f}}-s|<\gamma\}} - "
           r"\mathbf{1}_{\{|L^{\mathrm{dg}}-s|<\gamma\}}\right)$",
           r"$\frac{1}{\gamma}\left(\psi\,\left(\frac{L^{\mathrm{f}}-s}{\gamma}\right) - "
           r"\psi\,\left(\frac{L^{\mathrm{dg}}-s}{\gamma}\right)\right)$")


def _var_over_pairs(diff: np.ndarray) -> np.ndarray:
    """Sample variance across the coupled pairs (axis 0) at each grid point (axis 1)."""
    return diff.var(axis=0, ddof=1)


def _cdf_curves(s_eval: np.ndarray, Lf_c: np.ndarray, Ldg_c: np.ndarray, d_cdf: float,
                r_cdf: int) -> tuple[np.ndarray, np.ndarray]:
    """(unsmoothed indicator, GNR-smoothed) level-1 variances of the CDF difference at s_eval."""
    ind = (Lf_c <= s_eval).astype(float) - (Ldg_c <= s_eval).astype(float)
    g_sm = _g_for_cdf((Lf_c - s_eval) / d_cdf, r_cdf) - _g_for_cdf((Ldg_c - s_eval) / d_cdf, r_cdf)
    return _var_over_pairs(ind), _var_over_pairs(g_sm)


def _pdf_curves(s_eval: np.ndarray, Lf_c: np.ndarray, Ldg_c: np.ndarray, d_pdf: float,
                r_pdf: int) -> tuple[np.ndarray, np.ndarray]:
    """(boxcar, GNR-smoothed) level-1 variances of the PDF difference at s_eval."""
    box = ((np.abs(Lf_c - s_eval) < d_pdf).astype(float)
           - (np.abs(Ldg_c - s_eval) < d_pdf).astype(float)) / (2 * d_pdf)
    k_sm = (_g_for_pdf((Lf_c - s_eval) / d_pdf, r_pdf)
            - _g_for_pdf((Ldg_c - s_eval) / d_pdf, r_pdf)) / d_pdf
    return _var_over_pairs(box), _var_over_pairs(k_sm)


def _plot(xvals: np.ndarray, grey: np.ndarray, black: np.ndarray, labels: tuple[str, str],
          xlabel: str, fname: str, xlim: tuple[float, float],
          alpha_marks: tuple[float, ...] | None = None) -> None:
    """Plot the unsmoothed (grey) and GNR-smoothed (black) level-1 variance curves and save the PNG."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xvals, grey, color="0.6", lw=1.6, label=labels[0])
    ax.plot(xvals, black, color="black", lw=1.8, label=labels[1])
    if alpha_marks is not None:
        ymax = max(np.max(grey), np.max(black))
        for a in alpha_marks:
            ax.axvline(a, color="red", ls="--", lw=1.0, alpha=0.7)
            # label low in the (empty) lower region so it clears the legend and the grey curve
            ax.text(a, 0.04 * ymax, rf"$\alpha={a}$", rotation=90, color="red",
                    fontsize=10, va="bottom", ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(LBL_VAR)
    ax.set_xlim(*xlim)
    ax.legend()
    set_frame_height(fig)
    fig.savefig(str(RESULTS_DIR / fname), dpi=SAVE_DPI)
    plt.close(fig)


def main() -> None:
    """Compute the coupled level-1 variance curves and save the CDF/PDF figures vs s and vs u."""
    apply_style()

    # Coupled fine / delta-gamma losses on the SAME 50k scenarios (loss convention).
    full_pnl, _delta_pnl, dg_pnl, _val_date = get_full_delta_and_delta_gamma_pnl(N_SCEN)
    Lf, Ldg = -full_pnl, -dg_pnl

    # One BL-HD run to read the delta / k / r the adaptive algo converges to at C_BUDGET.
    sampler = portfolio_sampler()
    S_0, S_1, _ = calibrate_domain(sampler["fine"], n_pilot=200)
    np.random.seed(0)
    _q, diag = adapt_tuning_blhd_quantile(
        ALPHA, C_BUDGET, sampler["coarse"], sampler["level0"], sampler["level1"],
        C_F, C_DG, C_M, S_0, S_1, R, S_MH, K=K, rng=np.random.default_rng(0), verbose=False)
    d_cdf, r_cdf = diag["cdf_delta"], diag["cdf_r"]
    d_pdf, r_pdf = diag["pdf_delta"], diag["pdf_r"]
    print(f"CDF: delta={d_cdf:.4g} k={diag['cdf_k']} r={r_cdf}    "
          f"PDF: delta={d_pdf:.4g} k={diag['pdf_k']} r={r_pdf}")

    Lf_c, Ldg_c = Lf[:, None], Ldg[:, None]   # (N_SCEN, 1) for broadcasting against eval points
    Lf_sorted = np.sort(Lf)

    # option 1: vs loss level s over the whole domain, padded a bit beyond [S_0, S_1] so the tails
    # of the (near-zero) variance curves are visible rather than cut off exactly at the domain edge
    S_PAD = 0.15 * (S_1 - S_0)
    s_lo, s_hi = S_0 - S_PAD, S_1 + S_PAD
    s_grid = np.linspace(s_lo, s_hi, N_GRID)
    V_ind, V_sm = _cdf_curves(s_grid, Lf_c, Ldg_c, d_cdf, r_cdf)
    V_box, V_ksm = _pdf_curves(s_grid, Lf_c, Ldg_c, d_pdf, r_pdf)
    _plot(s_grid, V_ind, V_sm, LBL_CDF, r"$s$", "mlmc_level_variance_cdf.png", (s_lo, s_hi))
    _plot(s_grid, V_box, V_ksm, LBL_PDF, r"$s$", "mlmc_level_variance_pdf.png", (s_lo, s_hi))

    # option 2: vs quantile level u = F_{L^f}(s), tail region, with the alpha markers
    u_grid = np.linspace(U_LO, 0.999, N_GRID)
    s_of_u = np.quantile(Lf_sorted, u_grid)     # loss level at each quantile level
    Vu_ind, Vu_sm = _cdf_curves(s_of_u, Lf_c, Ldg_c, d_cdf, r_cdf)
    Vu_box, Vu_ksm = _pdf_curves(s_of_u, Lf_c, Ldg_c, d_pdf, r_pdf)
    xlab_u = r"$u = F_{L^{\mathrm{f}}}(s)$"
    _plot(u_grid, Vu_ind, Vu_sm, LBL_CDF, xlab_u, "mlmc_level_variance_cdf_vs_u.png",
          (U_LO, 1.0), alpha_marks=ALPHAS)
    _plot(u_grid, Vu_box, Vu_ksm, LBL_PDF, xlab_u, "mlmc_level_variance_pdf_vs_u.png",
          (U_LO, 1.0), alpha_marks=ALPHAS)

    # exact level-1 variance at each alpha (s = empirical alpha-quantile of L^f)
    print("level-1 variance at each alpha (CDF: indicator / smoothed ; PDF: box / smoothed):")
    for a in ALPHAS:
        s_a = float(np.quantile(Lf_sorted, a))
        ci, cs = _cdf_curves(np.array([s_a]), Lf_c, Ldg_c, d_cdf, r_cdf)
        pb, ps = _pdf_curves(np.array([s_a]), Lf_c, Ldg_c, d_pdf, r_pdf)
        print(f"  alpha={a:<5} s={s_a:8.2f}  CDF {ci[0]:.3e} / {cs[0]:.3e}   "
              f"PDF {pb[0]:.3e} / {ps[0]:.3e}")
    print("Saved 4 figures (cdf/pdf vs s and vs u)")


if __name__ == "__main__":
    main()

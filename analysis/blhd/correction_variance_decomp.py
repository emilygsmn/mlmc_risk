"""Variance decomposition of the bilevel BL-HD estimator vs total cost (portfolio, loss convention,
alpha=0.995). The estimator is q_hat = q_HD + correction, where q_HD is the level-0 Harrell-Davis
quantile of the N_dg delta-gamma samples and `correction` is the fine-level MH correction. Because
both are built from the SAME delta-gamma samples they are NOT independent (strong negative
covariance), so their variances do not add to the total -- that coupling is the estimator's built-in
control-variate mechanism.

For each cost point we run R_TRIALS independent BL-HD runs, record (q_HD, correction, q_hat) from the
diagnostics of each, and estimate:
  V[q_HD]              -- level-0 base variance,
  V[correction]        -- correction variance,
  V[q_hat]             -- total estimator variance,
  V[q_HD] + V[corr]    -- the "independent-levels" sum (what the total WOULD be with no coupling).
The gap between that sum and V[q_hat] is the variance reduction from the negative covariance.

Saves a JSON of the raw per-trial values + variances, and a log-log figure (house style, high-res)
to results/blhd/. Requires network access. This is a long job -- run it detached.
"""

import sys
import json
import warnings
from pathlib import Path

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
from mlmc_risk_estimation.blhd.plotting import (apply_style, SAVE_DPI, set_frame_height, LBL_COST,
                                               LBL_VAR)

ALPHA = 0.995
COST_GRID = list(np.geomspace(3e5, 1e7, 6))
R_TRIALS = 50
C_F, C_DG, C_M, S_MH, K, R = 210.0, 1.0, 0.70, 40.0, 20, 3
OUT = RESULTS_DIR / f"correction_variance_decomp_alpha{ALPHA}_ntrials{R_TRIALS}_npts{len(COST_GRID)}"


def _var_se(v: np.ndarray) -> tuple[float, float]:
    """Sample variance (ddof=1) and its approximate standard error, var*sqrt(2/(R-1))."""
    v = np.asarray(v, float)
    var = float(np.var(v, ddof=1))
    return var, var * np.sqrt(2.0 / (len(v) - 1))


def run() -> tuple[dict, dict]:
    """Run the variance-decomposition sweep over COST_GRID, save the raw results to JSON, and
    return (results, meta)."""
    apply_style()
    s = portfolio_sampler()
    S0, S1, _ = calibrate_domain(s["fine"], n_pilot=200)
    print(f"alpha={ALPHA}  domain=[{S0:.0f},{S1:.0f}]  R_TRIALS={R_TRIALS}  "
          f"costs={[f'{c:,.0f}' for c in COST_GRID]}")

    rng_master = np.random.default_rng(20260715)
    res = {"cost": [], "V_qHD": [], "V_qHD_se": [], "V_corr": [], "V_corr_se": [],
           "V_qhat": [], "V_qhat_se": [], "cov": [], "V_sum": [],
           "qHD": [], "corr": [], "qhat": []}
    for C in COST_GRID:
        qHD, corr, qhat = [], [], []
        for _ in range(R_TRIALS):
            seed = int(rng_master.integers(1 << 30))
            np.random.seed(seed)
            q, d = adapt_tuning_blhd_quantile(
                ALPHA, C, s["coarse"], s["level0"], s["level1"], C_F, C_DG, C_M, S0, S1, R, S_MH,
                K=K, rng=np.random.default_rng(seed), verbose=False)
            qHD.append(d["q_HD"]); corr.append(d["correction"]); qhat.append(q)
        v_qhd, se_qhd = _var_se(qHD)
        v_cor, se_cor = _var_se(corr)
        v_qht, se_qht = _var_se(qhat)
        cov = float(np.cov(qHD, corr, ddof=1)[0, 1])
        res["cost"].append(float(C))
        res["V_qHD"].append(v_qhd); res["V_qHD_se"].append(se_qhd)
        res["V_corr"].append(v_cor); res["V_corr_se"].append(se_cor)
        res["V_qhat"].append(v_qht); res["V_qhat_se"].append(se_qht)
        res["cov"].append(cov); res["V_sum"].append(v_qhd + v_cor)
        res["qHD"].append(qHD); res["corr"].append(corr); res["qhat"].append(qhat)
        print(f"C={C:>12,.0f}  V[q_HD]={v_qhd:.4f}  V[corr]={v_cor:.4f}  V[q_hat]={v_qht:.4f}  "
              f"2cov={2*cov:+.4f}  sum={v_qhd+v_cor:.4f}")

    meta = dict(alpha=ALPHA, c_f=C_F, r_trials=R_TRIALS, cost_grid=COST_GRID)
    json.dump({"results": res, "meta": meta}, open(str(OUT) + ".json", "w"), indent=2)
    return res, meta


def plot(res: dict, meta: dict) -> None:
    """Plot the three variance curves and the no-coupling sum vs cost, and save the figure."""
    apply_style()
    cf = meta["c_f"]
    x = np.array(res["cost"]) / cf
    fig, ax = plt.subplots(figsize=(6, 6))
    # Three variance curves + the no-coupling sum (dashed). All BL-HD components -> black family.
    ax.errorbar(x, res["V_qhat"], yerr=res["V_qhat_se"], color="black", lw=1.9, marker="o",
                markersize=6, markerfacecolor="none", capsize=3,
                label=r"$\mathbb{V}[\widehat{q}^{\ \mathrm{BL\text{-}HD}}]$")
    ax.errorbar(x, res["V_corr"], yerr=res["V_corr_se"], color="0.4", lw=1.7, marker="^",
                markersize=6, markerfacecolor="none", capsize=3,
                label=r"$\mathbb{V}[\widehat{c}]$")
    ax.errorbar(x, res["V_qHD"], yerr=res["V_qHD_se"], color="0.65", lw=1.7, marker="s",
                markersize=6, markerfacecolor="none", capsize=3,
                label=r"$\mathbb{V}[\widehat{q}_{\mathrm{HD}}]$")
    ax.plot(x, res["V_sum"], color="black", linestyle=":", lw=1.4,
            label=r"$\mathbb{V}[\widehat{q}_{\mathrm{HD}}]+\mathbb{V}[\widehat{c}]$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(LBL_COST)
    ax.set_ylabel(LBL_VAR)
    ax.legend()
    set_frame_height(fig)
    fig.savefig(str(OUT) + ".png", dpi=SAVE_DPI)
    plt.close(fig)
    print("Saved", str(OUT) + ".png")


if __name__ == "__main__":
    r, m = run()
    plot(r, m)
    print("ALL DONE")

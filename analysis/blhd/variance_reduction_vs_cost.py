"""Variance-reduction factor (standard OS variance / BL-HD variance) vs total cost, at alpha=0.995,
for many model x correlation combinations on one plot. Factor per budget = (OS_fine_rmse / BL-HD_rmse)^2
(alpha=0.995 is variance-limited, so RMSE^2 ~ variance). Analytical models (normal, t5, t10, t15) are
read from the vary-rho JSONs (rho in {0.5,0.9,0.99,0.999}); the portfolio is its own fixed empirical
coupling. Colour = model, line style = rho; a grey line at factor=1 marks the BL-HD break-even.
PNG -> results/blhd/. Reads existing JSONs only (no recompute)."""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from _common import RESULTS_DIR
from mlmc_risk_estimation.blhd.plotting import apply_style, SAVE_DPI, set_frame_height, LBL_COST

MODEL_COLOR = {"normal": "#1f77b4", "t5": "#d62728", "t10": "#2ca02c", "t15": "#9467bd"}
MODEL_LABEL = {"normal": r"$\mathcal{N}(0,1)$", "t5": r"$t_5$", "t10": r"$t_{10}$", "t15": r"$t_{15}$"}
RHO_STYLE = {"0.5": ":", "0.9": "-.", "0.99": "--", "0.999": "-"}


def _factor(bl: dict[str, list[float]], base: dict[str, list[float]]) -> np.ndarray:
    """Variance-reduction factor (base_rmse/bl_rmse)^2 between two arms' result dicts."""
    return (np.array(base["rmse"]) / np.array(bl["rmse"])) ** 2


def main() -> None:
    """Plot the OS/BL-HD variance-reduction factor vs. cost for every model x rho combination."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 6))

    # analytical: normal + student-t, coloured by model, styled by rho
    files = (sorted(glob.glob(str(RESULTS_DIR / "rmse_vs_cost_modelnormal_*alpha0.995*.json")))
             + sorted(glob.glob(str(RESULTS_DIR / "rmse_vs_cost_modelstudentt_*alpha0.995*.json"))))
    for f in files:
        d = json.load(open(f)); res = d["results"]; cf = d["meta"].get("c_f", 250.0)
        model = "normal" if "normal" in f else "t" + f.split("dof")[1].split("_")[0]
        base = res["OS (fine)"]
        for key in res:
            if not key.startswith("BL-HD (rho="):
                continue
            rho = key.split("rho=")[1].rstrip(")")
            x = np.array(res[key]["cost"]) / cf
            ax.plot(x, _factor(res[key], base), color=MODEL_COLOR[model],
                    linestyle=RHO_STYLE[rho], lw=1.4)

    # portfolio: fixed empirical coupling, black
    for f in sorted(glob.glob(str(RESULTS_DIR / "rmse_vs_cost_modelportfolio_alpha0.995*.json"))):
        d = json.load(open(f)); res = d["results"]; cf = d["meta"].get("c_f", 210.0)
        if "BL-HD" not in res or "OS (fine)" not in res:
            continue
        x = np.array(res["BL-HD"]["cost"]) / cf
        ax.plot(x, _factor(res["BL-HD"], res["OS (fine)"]), color="black", lw=2.2,
                marker="o", markersize=5, markerfacecolor="none", zorder=5)

    ax.axhline(1.0, color="0.5", linestyle="--", lw=1.0)   # BL-HD break-even
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(LBL_COST)
    ax.set_ylabel(r"variance-reduction factor  $\mathbb{V}_{\mathrm{OS}}/\mathbb{V}_{\mathrm{BL\text{-}HD}}$")

    model_handles = [Line2D([0], [0], color=MODEL_COLOR[m], lw=2, label=MODEL_LABEL[m])
                     for m in ("normal", "t5", "t10", "t15")]
    model_handles.append(Line2D([0], [0], color="black", lw=2.2, marker="o", markerfacecolor="none",
                                label="portfolio (DG)"))
    rho_handles = [Line2D([0], [0], color="0.35", linestyle=RHO_STYLE[r], lw=1.6,
                          label=rf"$\varrho={r}$") for r in ("0.5", "0.9", "0.99", "0.999")]
    leg1 = ax.legend(handles=model_handles, title="model", loc="upper left", fontsize=10)
    ax.add_artist(leg1)
    ax.legend(handles=rho_handles, title=r"corr. $\varrho$", loc="lower right", fontsize=10)

    set_frame_height(fig)
    path = str(RESULTS_DIR / "variance_reduction_vs_cost.png")
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)
    print("Saved", path)


if __name__ == "__main__":
    main()

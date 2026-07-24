"""Trace of the final production Metropolis-Hastings chain the BL-HD estimator draws from the tilted
target density pi_hat (portfolio, loss convention, alpha=0.995). Runs the estimator once, pulls the
production chain states from the diagnostics, and plots the trace (black) with the pseudo-true
quantile (dashed red) and the final BL-HD estimate (solid grey) marked as horizontal lines. A faint
grey band of +/- 1 RMSE around the reference -- the typical BL-HD estimate spread at that budget,
from N_RMSE_SEEDS independent runs -- shows that the single grey line is one draw within the expected
scatter, and that the band narrows with budget. Also saves, per budget, an autocorrelation plot of
the chain (mh_acf_*.png) and the mixing statistics (acceptance rate, integrated autocorrelation time
tau, ESS = N/tau, lag-1 ACF) to results/blhd/mh_chain_stats.csv.

Wide format (12, 4.5), house style, high-res. Requires network access.

Run:  ./venv/bin/python analysis/blhd/mh_trace.py
It regenerates every mh_trace_*.png and mh_acf_*.png and (over)writes mh_chain_stats.csv. The
RMSE band is the slow part (N_RMSE_SEEDS=50 estimator runs per budget, incl. the 1e7 runs); the
trace/ACF/stats themselves only need one seed-0 chain per budget. Lower N_RMSE_SEEDS for a quick run.
"""

import csv
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

from mlmc_risk_estimation.blhd.samplers.portfolio import portfolio_sampler
from mlmc_risk_estimation.blhd.experiment import calibrate_domain, estimate_pseudo_true_quantile
from mlmc_risk_estimation.blhd.adaptive import adapt_tuning_blhd_quantile
from mlmc_risk_estimation.blhd.plotting import apply_style, SAVE_DPI, build_filename_tag, set_frame_height

ALPHA = 0.995
BUDGETS = (1e4, 1e5, 1e6, 1e7)
N_PSEUDO = 500_000    # sample size for the OS pseudo-true VaR reference
N_RMSE_SEEDS = 50     # independent runs per budget for the +/- RMSE band (matches the RMSE-vs-cost n_trials)
C_F, C_DG, C_M, S_MH, K, R = 210.0, 1.0, 0.70, 40.0, 20, 3


def _estimate(sampler: dict, S_0: float, S_1: float, C_star: float, seed: int) -> tuple[float, dict]:
    """One BL-HD run at the given seed; returns (q_hat, diagnostics)."""
    np.random.seed(seed)
    return adapt_tuning_blhd_quantile(
        ALPHA, C_star, sampler["coarse"], sampler["level0"], sampler["level1"],
        C_F, C_DG, C_M, S_0, S_1, R, S_MH, K=K, rng=np.random.default_rng(seed), verbose=False)


def _rmse_and_stats_for_budget(sampler: dict, S_0: float, S_1: float, true_q: float, C_star: float,
                              rng_master: np.random.Generator) -> tuple[float, dict]:
    """Over N_RMSE_SEEDS independent runs at this budget: the RMSE of the BL-HD estimate, plus the
    mean and (sample) std across those runs of the chain mixing statistics AR, IAT tau and ESS.
    Each run's mixing statistic is a noisy single-chain estimate; averaging over the runs gives the
    stable sampler-level value (esp. for tau/ESS). Returns (rmse, agg_dict)."""
    errs, ar, tau, ess = [], [], [], []
    for _ in range(N_RMSE_SEEDS):
        seed = int(rng_master.integers(1 << 30))
        q, diag = _estimate(sampler, S_0, S_1, C_star, seed)
        errs.append(q - true_q)
        st = _chain_stats(np.asarray(diag["prod_samples"]), diag)
        ar.append(st["acceptance_rate"]); tau.append(st["iat_tau"]); ess.append(st["ess"])
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    root_n = np.sqrt(N_RMSE_SEEDS)

    def _ms(v: list[float]) -> tuple[float, float, float]:
        """Mean, sample std (ddof=1), standard error (std/sqrt(n))."""
        std = float(np.std(v, ddof=1))
        return float(np.mean(v)), std, std / root_n

    ar_m, ar_s, ar_se = _ms(ar)
    tau_m, tau_s, tau_se = _ms(tau)
    ess_m, ess_s, ess_se = _ms(ess)
    agg = dict(
        acceptance_rate_mean=ar_m, acceptance_rate_std=ar_s, acceptance_rate_se=ar_se,
        iat_tau_mean=tau_m,        iat_tau_std=tau_s,        iat_tau_se=tau_se,
        ess_mean=ess_m,            ess_std=ess_s,            ess_se=ess_se,
    )
    return rmse, agg


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation rho_k for k = 0..max_lag."""
    x = np.asarray(x, float) - np.mean(x)
    denom = np.dot(x, x)
    return np.array([1.0 if k == 0 else np.dot(x[:-k], x[k:]) / denom
                     for k in range(max_lag + 1)])


def _chain_stats(chain: np.ndarray, diag: dict) -> dict:
    """Mixing statistics of the production chain: acceptance rate (AR), integrated autocorrelation
    time (IAT tau, summed until the first non-positive autocorrelation), effective sample size
    ESS = N / tau, and lag-1 autocorrelation."""
    N = chain.size
    acf = _acf(chain, min(N - 1, 200))
    tau = 1.0
    for k in range(1, len(acf)):
        if acf[k] <= 0:
            break
        tau += 2.0 * acf[k]
    return dict(N=N,
                acceptance_rate=float(diag["production_mh_acc_rate"]),
                lag1_acf=float(acf[1]) if N > 1 else float("nan"),
                iat_tau=float(tau),
                ess=float(N / tau),
                ess_frac=float(1.0 / tau),
                s_mh=float(diag["s_mh_final"]))


def _plot_acf(chain: np.ndarray, C_star: float, max_lag: int = 50) -> None:
    """Autocorrelation of the (seed-0) production chain vs lag, same size as the trace plots.
    Marks the +/- 1.96/sqrt(N^m) white-noise band, where N^m is the MH (production chain) length."""
    acf = _acf(chain, max_lag)
    lags = np.arange(max_lag + 1)
    ci = 1.96 / np.sqrt(chain.size)   # chain.size = N^m, the production MH sample count

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axhspan(-ci, ci, color="red", alpha=0.13, lw=0,
               label=r"$\pm 1.96/\sqrt{N^{\mathrm{m}}}$")
    # Unfilled (outline-only) bars, full width so adjacent bars touch (no gaps); the red band
    # stays visible through them.
    ax.bar(lags, acf, width=1.0, fill=False, edgecolor="black", linewidth=1.0, zorder=2)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\widehat{\rho}_k$")
    ax.set_xlim(-0.5, max_lag + 0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    set_frame_height(fig)

    tag = build_filename_tag(model="portfolio", alpha=ALPHA, C=int(C_star))
    path = str(RESULTS_DIR / f"mh_acf_{tag}.png")
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_for_budget(sampler: dict, S_0: float, S_1: float, true_q: float, C_star: float,
                     rmse: float, agg: dict) -> dict:
    """Plot the production chain trace and ACF for one budget, and return its row of stats."""
    q_hat, diag = _estimate(sampler, S_0, S_1, C_star, 0)   # seed-0 chain is the one shown
    chain = np.asarray(diag["prod_samples"])
    stats = _chain_stats(chain, diag)   # seed-0 (shown-chain) mixing statistics
    print(f"C={C_star:.0e}  N={stats['N']}  "
          f"AR seed0={stats['acceptance_rate']:.3f} mean={agg['acceptance_rate_mean']:.3f}  "
          f"tau seed0={stats['iat_tau']:.2f} mean={agg['iat_tau_mean']:.2f}  "
          f"ESS seed0={stats['ess']:.0f} mean={agg['ess_mean']:.0f}  RMSE={rmse:.3f}")
    _plot_acf(chain, C_star)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(np.arange(chain.size), chain, color="black", lw=0.7, zorder=1)
    # +/- 1 RMSE band around the reference: typical BL-HD estimate spread at this budget. Drawn on
    # top of the (much wider) per-sample chain envelope, tinted red and with dotted edges so the
    # thin band stays legible against the dense black trace.
    ax.axhspan(true_q - rmse, true_q + rmse, color="red", alpha=0.13, lw=0, zorder=2,
               label=r"$\widehat{\mathrm{VaR}}_{\alpha}^{\ \mathrm{OS}} \pm \mathrm{RMSE}$")
    for edge in (true_q - rmse, true_q + rmse):
        ax.axhline(edge, color="red", linestyle=":", lw=0.9, alpha=0.6, zorder=2)
    n_exp = int(round(np.log10(N_PSEUDO)))
    n_mant = N_PSEUDO / 10 ** n_exp
    n_str = (rf"10^{{{n_exp}}}" if abs(n_mant - 1.0) < 1e-9
             else rf"{n_mant:g}\times 10^{{{n_exp}}}")
    ax.axhline(true_q, color="red", linestyle="--", lw=1.5, zorder=3,
               label=rf"$\widehat{{\mathrm{{VaR}}}}_{{\alpha}}^{{\ \mathrm{{OS}}}}$ ($n={n_str}$)")
    ax.axhline(q_hat, color="0.5", linestyle="-", lw=1.5, zorder=3,
               label=r"$\widehat{\mathrm{VaR}}_{\alpha}^{\ \mathrm{BL\text{-}HD}}$")
    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$P_i$")
    ax.set_xlim(0, chain.size - 1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])   # inverted order: BL-HD, OS, band
    set_frame_height(fig)

    tag = build_filename_tag(model="portfolio", alpha=ALPHA, C=int(C_star))
    path = str(RESULTS_DIR / f"mh_trace_{tag}.png")
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"Saved {path}")

    return dict(
        C_total=float(C_star), N=stats["N"],
        acceptance_rate_seed0=stats["acceptance_rate"], acceptance_rate_mean=agg["acceptance_rate_mean"],
        acceptance_rate_std=agg["acceptance_rate_std"], acceptance_rate_se=agg["acceptance_rate_se"],
        iat_tau_seed0=stats["iat_tau"], iat_tau_mean=agg["iat_tau_mean"],
        iat_tau_std=agg["iat_tau_std"], iat_tau_se=agg["iat_tau_se"],
        ess_seed0=stats["ess"], ess_mean=agg["ess_mean"],
        ess_std=agg["ess_std"], ess_se=agg["ess_se"],
        lag1_acf_seed0=stats["lag1_acf"], s_mh_seed0=stats["s_mh"],
        q_hat=float(q_hat), true_q=float(true_q), rmse=float(rmse), n_rmse_seeds=N_RMSE_SEEDS)


def _write_stats_csv(rows: list[dict]) -> None:
    """Write the per-budget mixing statistics to a CSV in results/blhd/. For AR, IAT (tau) and ESS
    the CSV holds the seed-0 (shown-chain) value plus the mean, sample-std (ddof=1) and standard
    error (std/sqrt(N_RMSE_SEEDS)) over the N_RMSE_SEEDS runs; lag1_acf and s_mh are seed-0 only."""
    path = RESULTS_DIR / "mh_chain_stats.csv"
    cols = ["C_total", "N",
            "acceptance_rate_seed0", "acceptance_rate_mean", "acceptance_rate_std", "acceptance_rate_se",
            "iat_tau_seed0", "iat_tau_mean", "iat_tau_std", "iat_tau_se",
            "ess_seed0", "ess_mean", "ess_std", "ess_se",
            "lag1_acf_seed0", "s_mh_seed0", "q_hat", "true_q", "rmse", "n_rmse_seeds"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"Saved {path}")


def main() -> None:
    """Run the trace/ACF/RMSE-band analysis for each budget in BUDGETS and write the stats CSV."""
    apply_style()
    sampler = portfolio_sampler()
    S_0, S_1, _ = calibrate_domain(sampler["fine"], n_pilot=200)
    true_q = estimate_pseudo_true_quantile(sampler["fine"], ALPHA, N_PSEUDO, seed=0)
    rng_master = np.random.default_rng(20260715)
    rows = []
    for C_star in BUDGETS:
        rmse, agg = _rmse_and_stats_for_budget(sampler, S_0, S_1, true_q, C_star, rng_master)
        rows.append(_plot_for_budget(sampler, S_0, S_1, true_q, C_star, rmse, agg))
    _write_stats_csv(rows)


if __name__ == "__main__":
    main()

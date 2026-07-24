"""
Main loop: AdaptTuningBlHdQuantile.

This is the adaptive tuning heuristic for the Bilevel Harrell-Davis estimator.
It is built from:

    _dg_estimators.py   :   Resamples delta_gamma
    _adapt_step.py      :   Performs an adaptive cost step
    _warm_mlmc.py       :   Computes Giles' MLMC CDF/PDF (adjusted for warm-starting)
    _mh_sampler.py      :   Builds beta-tilted density, performs Metropolis-Hastings
    _marginal_gains.py  :   Evaluates the marginal gains of DG, CDF, PDF

Main idea:
  - The 3 components (DG, CDF, PDF) are initialized to set up an initial
    estimate.
  - Each of DG/CDF/PDF maintains a "before"/"after" state of its own axis,
    taken at the time of its own last actual investment.
  - For the first marginal gain evaluation, after init, one C_step-sized
    cost investment needs to be made in each of the three, to produce
    the first "before"/"after" pair for each.
  - Every iteration, all three gains are (cheaply) re-evaluated via
    evaluate_gain, which reweights the pilot sample under these existing
    states and the current shared context.
  - Only the winner gets a new investment: its "after" state is replaced
    with a fresh one. This incurs cost. Its "before" state becomes what
    its "after" state used to be.
"""

from collections.abc import Callable

import numpy as np
from scipy.interpolate import interp1d

from mlmc_risk_estimation.blhd.baselines import harrell_davis, order_statistic_quantile
from mlmc_risk_estimation.blhd.gnr import build_monotone_cdf_interp, build_nonneg_pdf_interp
from mlmc_risk_estimation.blhd.mlmc_df import (warm_adaptive_mlmc, warm_adaptive_mlmc_pdf,
                                               initial_state_cost)
from mlmc_risk_estimation.blhd.mh import (build_log_pi_hat_from_callables,
                                          grw_metropolis_hastings, tune_step_size)
from mlmc_risk_estimation.blhd.marginal_gains import evaluate_gain

__all__ = ["adapt_tuning_blhd_quantile", "adapt_step_cost"]

# init_delta_gamma, resample_delta_gamma, build_empirical_qf (DG-state machinery) and
# adapt_step_cost are defined lower in this module.


def _interp_fn(s_grid: np.ndarray, vals: np.ndarray, fill_left: float, fill_right: float,
              clip_low: float | None = None,
              clip_high: float | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """Build a vectorized interpolant from an MLMC output grid."""

    f = interp1d(s_grid, vals, kind=3, bounds_error=False,
                 fill_value=(fill_left, fill_right))

    def evaluate(x: np.ndarray) -> np.ndarray:
        """Evaluate the clipped cubic interpolant at x."""
        return np.clip(f(x), clip_low, clip_high)

    return evaluate


def adapt_tuning_blhd_quantile(
        alpha: float, C_star: float,
        dg_sampler: Callable[[int], np.ndarray],
        lvl0_f: Callable, lvl1_f: Callable,     # fine-model level0/level1 samplers
        c_f: float, c_dg: float, c_m: float,
        S_0: float, S_1: float, r: int,
        s_mh: float,
        K: int = 40, c_dg_unit: float | None = None,
        allocation: str = "adaptive",
        init_cost_frac: float = 0.1,
        eps_hint_cdf: float = 0.1, eps_hint_pdf: float = 0.1,
        adapt_step_cost_tol: float = 0.15, adapt_step_max_probes: int = 30,
        mh_burn_in_frac: float = 0.2,
        ess_refresh_frac: float = 0.3,
        gain_method: str = "original", n_boot: int = 25,
        tune_s_mh: bool = True, s_mh_target_ar: float = 0.44, s_mh_ar_tol: float = 0.03,
        s_mh_pilot_n: int = 2000,
        rng: np.random.Generator | None = None, verbose: bool = True) -> tuple[float, dict]:
    """
    AdaptTuningBlHdQuantile: bilevel Harrell-Davis + MLMC-correction
    quantile estimator with adaptive budget allocation.

    Every MH chain (pilot, every pilot refresh, production) self-starts
    from the current Harrell-Davis DG estimate; see _find_valid_start.
    There is no external starting-point parameter.
    """
    if rng is None:
        rng = np.random.default_rng()

    # DG chunks are sized by c_dg (n_dg_step = C_step / c_dg) but blled at c_dg_unit per sample
    if c_dg_unit is None:
        c_dg_unit = c_dg

    # Cost per fine-model evaluation, charged against the (absolute) budget
    cost_M = c_f
    diagnostics = dict(iterations=[], warnings=[], pilot_refreshes=[])

    # Budget split
    C_MH = C_star / 100.0
    C_step = (C_star - C_MH) / K
    C_res = C_star - C_MH
    N_m_p = int(np.floor(C_MH / (100.0 * c_m)))
    N_m = int(np.floor(99.0 * C_MH / (100.0 * c_m)))

    if verbose:
        print(f"Budget: C*={C_star:.0f}  C_MH={C_MH:.0f}  C_step={C_step:.1f}  "
              f"C_res={C_res:.0f}  N_m_p={N_m_p}  N_m={N_m}")

    r_cdf = r + 1

    # Initialisation: One minimal DG draw + one MLMC run each for CDF/PDF.
    # No gain computed as there is no valid "smaller" state to compare to.

    # AdaptInit's CDF/PDF cost is targeted at init_cost_frac * C_step
    # (via adapt_step_cost) rather than a fixed error tolerance.
    n_dg_floor = int(np.ceil(2.0 / min(alpha, 1 - alpha) - 1))
    dg_state = init_delta_gamma(dg_sampler, n_dg_floor, alpha, c_dg=c_dg_unit)

    # Any fresh warm_adaptive_mlmc(_pdf) call has a hard cost floor from
    # its mandatory 100+100 seed samples (N_init=100, hardcoded in
    # _warm_mlmc.py's _fresh_state), regardless of eps:
    #   floor = 100 * c_dg + 100 * cost_M
    # Setting C_init_target below this floor is unreachable by construction.
    mlmc_seed_floor = 100.0 * c_dg_unit + 100.0 * cost_M
    C_init_target = max(init_cost_frac * C_step, 1.05 * mlmc_seed_floor)

    s_cdf, F_vals, state_CDF, ic_cdf_init = adapt_step_cost(
        None, C_init_target, S_0, S_1, r_cdf, cost_M, lvl0_f, lvl1_f,
        warm_adaptive_mlmc, eps_hint=eps_hint_cdf,
        cost_tol=adapt_step_cost_tol, max_probes=adapt_step_max_probes)
    C_CDF = ic_cdf_init["cost_total"]

    s_pdf, rho_vals, state_PDF, ic_pdf_init = adapt_step_cost(
        None, C_init_target, S_0, S_1, r, cost_M, lvl0_f, lvl1_f,
        warm_adaptive_mlmc_pdf, eps_hint=eps_hint_pdf,
        cost_tol=adapt_step_cost_tol, max_probes=adapt_step_max_probes)
    C_PDF = ic_pdf_init["cost_total"]

    # Ensure monotonicity of CDF (see Giles, Nagapetyan, Ritter 2017)
    F_hat = build_monotone_cdf_interp(s_cdf, F_vals, r_cdf)
    # Ensure non-negativity of PDF (see Giles, Nagapetyan, Ritter 2015)
    rho_hat = build_nonneg_pdf_interp(s_pdf, rho_vals, r)

    C_DG = dg_state["C_DG"]

    if verbose:
        print(f"Init: N_dg={dg_state['N_dg']}  C_DG={C_DG:.0f}  "
              f"C_CDF_init={C_CDF:.0f}  C_PDF_init={C_PDF:.0f}  "
              f"(target was {C_init_target:.0f})")

    # Draw a pilot chain from the init state
    def _find_valid_start(log_pi: Callable[[np.ndarray], np.ndarray], candidates: list[float],
                          grid_n: int = 201) -> float:
        """Return the first candidate with finite log_pi, or else the best point on a scan grid."""
        for c in candidates:
            if c is not None and np.isfinite(log_pi(c)):
                return c
        grid = np.linspace(S_0, S_1, grid_n)
        vals = log_pi(grid)
        finite = np.isfinite(vals)
        if not np.any(finite):
            raise RuntimeError(
                "No point in [S_0, S_1] has finite log_pi_hat under the "
                "current (a, b, F_hat, rho_hat), F_hat/rho_hat have "
                "degenerated across the whole domain, not just one point")
        return grid[finite][np.argmax(vals[finite])]

    current_s_mh = s_mh  # warm-started across every _draw_pilot/production call

    def _draw_pilot(a0: float, b0: float, F_hat0: Callable, rho_hat0: Callable,
                    start_x_candidates: list[float]) -> tuple[Callable, np.ndarray, float]:
        """Build the tilted target log_pi_0 from (a0, b0, F_hat0, rho_hat0) and draw a fresh pilot
        MH chain from it, re-tuning the proposal step first if tune_s_mh."""
        nonlocal current_s_mh
        log_pi_0 = build_log_pi_hat_from_callables(a0, b0, F_hat0, rho_hat0, S_0, S_1)
        start_x = _find_valid_start(log_pi_0, start_x_candidates)
        if tune_s_mh:
            current_s_mh, tuned_ar = tune_step_size(
                log_pi_0, start_x, initial_guess=current_s_mh,
                target_ar=s_mh_target_ar, ar_tol=s_mh_ar_tol,
                pilot_n=s_mh_pilot_n, rng=rng)
        samples, acc = grw_metropolis_hastings(
            log_pi_0, start_x, current_s_mh, N_m_p,
            burn_in=int(mh_burn_in_frac * N_m_p), rng=rng)
        return log_pi_0, samples, acc

    a0, b0 = dg_state["a"], dg_state["b"]
    F_hat0, rho_hat0 = F_hat, rho_hat
    log_pi_0, pilot_samples, pilot_acc = _draw_pilot(a0, b0, F_hat0, rho_hat0, [dg_state["q_hd"]])
    if verbose:
        print(f"Pilot MH: N={N_m_p}  acc_rate={pilot_acc:.3f}  s_mh={current_s_mh:.5g}  "
              f"start=q_hd={dg_state['q_hd']:.4f}")

    def _shared(exclude: str) -> dict:
        """Current shared context, excluding whichever axis is varying."""
        return dict(
            a_shared=dg_state["a"], b_shared=dg_state["b"],
            F_hat_shared=F_hat, rho_hat_shared=rho_hat,
            F_inv_dg_shared=dg_state["F_inv_dg"], q_hd_shared=dg_state["q_hd"],
        )

    def _gain(axis: str, before: object, after: object, cost_last: float,
             state_before: dict | None = None, state_after: dict | None = None) -> tuple[float, float]:
        """Marginal gain per unit cost for one component, via importance-sampling reweighting
        (evaluate_gain).
        
        Returns (gain, ess) so the pilot-refresh mechanism can use the ESS."""

        if gain_method != "original":
            raise NotImplementedError(
                "only gain_method='original' is supported; the 'directional' bootstrap path "
                "(_bootstrap_gains) was experimental and is not part of the package")
        g, ess_b, ess_a = evaluate_gain(
            pilot_samples, log_pi_0, S_0, S_1,
            axis=axis, before=before, after=after, cost_last=cost_last,
            **_shared(axis))
        return g, min(ess_b, ess_a)

    # One C_step-sized investment in each of the three components, to establish
    # the first real before/after pair for each.
    n_dg_step = max(int(np.floor(C_step / c_dg)), 1)
    dg_state_before = dg_state
    dg_before = (dg_state["a"], dg_state["b"], dg_state["F_inv_dg"], dg_state["q_hd"])
    dg_state = resample_delta_gamma(dg_state, n_dg_step, dg_sampler, alpha, c_dg=c_dg_unit)
    # Deduct cost of this bootstrap investment
    C_DG_last = dg_state["C_DG"] - (C_DG)
    C_DG = dg_state["C_DG"]
    dg_after = (dg_state["a"], dg_state["b"], dg_state["F_inv_dg"], dg_state["q_hd"])
    G_DG, ess_dg = _gain("DG", dg_before, dg_after, C_DG_last, dg_state_before, dg_state)

    F_hat_before = F_hat
    state_CDF_before = state_CDF
    s_cdf, F_vals, state_CDF, ic_cdf = adapt_step_cost(
        state_CDF, C_step, S_0, S_1, r_cdf, cost_M, lvl0_f, lvl1_f,
        warm_adaptive_mlmc, eps_hint=eps_hint_cdf,
        cost_tol=adapt_step_cost_tol, max_probes=adapt_step_max_probes)
    F_hat = build_monotone_cdf_interp(s_cdf, F_vals, r_cdf)
    C_CDF_last = ic_cdf["cost_total"]
    C_CDF += C_CDF_last
    G_CDF, ess_cdf = _gain("CDF", F_hat_before, F_hat, C_CDF_last, state_CDF_before, state_CDF)

    rho_hat_before = rho_hat
    state_PDF_before = state_PDF
    s_pdf, rho_vals, state_PDF, ic_pdf = adapt_step_cost(
        state_PDF, C_step, S_0, S_1, r, cost_M, lvl0_f, lvl1_f,
        warm_adaptive_mlmc_pdf, eps_hint=eps_hint_pdf,
        cost_tol=adapt_step_cost_tol, max_probes=adapt_step_max_probes)
    rho_hat = build_nonneg_pdf_interp(s_pdf, rho_vals, r)
    C_PDF_last = ic_pdf["cost_total"]
    C_PDF += C_PDF_last
    G_PDF, ess_pdf = _gain("PDF", rho_hat_before, rho_hat, C_PDF_last, state_PDF_before, state_PDF)

    if verbose:
        print(f"Bootstrap: G_DG={G_DG:.3e}  G_CDF={G_CDF:.3e}  G_PDF={G_PDF:.3e}  "
              f"C_DG={C_DG:.0f}  C_CDF={C_CDF:.0f}  C_PDF={C_PDF:.0f}  "
              f"[gain_method={gain_method}]")

    # Marginal gains loop: Only the winner gets a new investment and the
    # other two gains are cheaply re-evaluated against their existing
    # before/after states under the (possibly-changed) shared context.
    it = 0
    while C_DG + C_CDF + C_PDF < C_res:
        it += 1

        if allocation == "naive":
            # Naive equal-split baseline: Split the C_step chunks across the three
            # components regardless of marginal gain (DG -> CDF -> PDF -> ...).
            # The standard-MLMC analog of splitting the budget/tolerance evenly
            # instead of trying to optimize the weighting.
            winner = ("DG", "CDF", "PDF")[(it - 1) % 3]
        else:
            winner = max(("DG", G_DG), ("CDF", G_CDF), ("PDF", G_PDF), key=lambda t: t[1])[0]

        if winner == "DG":
            dg_state_before = dg_state
            dg_before = dg_after
            dg_state = resample_delta_gamma(dg_state, n_dg_step, dg_sampler, alpha, c_dg=c_dg_unit)
            C_DG_last = dg_state["C_DG"] - C_DG
            C_DG = dg_state["C_DG"]
            dg_after = (dg_state["a"], dg_state["b"], dg_state["F_inv_dg"], dg_state["q_hd"])
            G_DG, ess_dg = _gain("DG", dg_before, dg_after, C_DG_last, dg_state_before, dg_state)

        elif winner == "CDF":
            F_hat_before = F_hat
            state_CDF_before = state_CDF
            s_cdf, F_vals, state_CDF, ic_cdf = adapt_step_cost(
                state_CDF, C_step, S_0, S_1, r_cdf, cost_M, lvl0_f, lvl1_f,
                warm_adaptive_mlmc, eps_hint=eps_hint_cdf,
                cost_tol=adapt_step_cost_tol, max_probes=adapt_step_max_probes)
            F_hat = build_monotone_cdf_interp(s_cdf, F_vals, r_cdf)
            C_CDF_last = ic_cdf["cost_total"]
            C_CDF += C_CDF_last
            G_CDF, ess_cdf = _gain("CDF", F_hat_before, F_hat, C_CDF_last, state_CDF_before, state_CDF)

        else:  # PDF
            rho_hat_before = rho_hat
            state_PDF_before = state_PDF
            s_pdf, rho_vals, state_PDF, ic_pdf = adapt_step_cost(
                state_PDF, C_step, S_0, S_1, r, cost_M, lvl0_f, lvl1_f,
                warm_adaptive_mlmc_pdf, eps_hint=eps_hint_pdf,
                cost_tol=adapt_step_cost_tol, max_probes=adapt_step_max_probes)
            rho_hat = build_nonneg_pdf_interp(s_pdf, rho_vals, r)
            C_PDF_last = ic_pdf["cost_total"]
            C_PDF += C_PDF_last
            G_PDF, ess_pdf = _gain("PDF", rho_hat_before, rho_hat, C_PDF_last, state_PDF_before, state_PDF)

        # Re-evaluate the other two gains under the (possibly-changed) shared context, using
        # their existing before/after states.
        if winner != "DG":
            G_DG, ess_dg = _gain("DG", dg_before, dg_after, C_DG_last, dg_state_before, dg_state)
        if winner != "CDF":
            G_CDF, ess_cdf = _gain("CDF", F_hat_before, F_hat, C_CDF_last, state_CDF_before, state_CDF)
        if winner != "PDF":
            G_PDF, ess_pdf = _gain("PDF", rho_hat_before, rho_hat, C_PDF_last, state_PDF_before, state_PDF)

        ess_min = min(ess_dg, ess_cdf, ess_pdf)
        diagnostics["iterations"].append(dict(
            it=it, winner=winner,
            G_DG=G_DG, G_CDF=G_CDF, G_PDF=G_PDF,
            C_DG=C_DG, C_CDF=C_CDF, C_PDF=C_PDF,
            ess_min=ess_min,
        ))
        if verbose and (it % 5 == 0 or it == 1):
            print(f"  it={it:3d}  winner={winner:4s}  "
                  f"G_DG={G_DG:.3e}  G_CDF={G_CDF:.3e}  G_PDF={G_PDF:.3e}  "
                  f"C_DG={C_DG:.0f}  C_CDF={C_CDF:.0f}  C_PDF={C_PDF:.0f}  "
                  f"ess_min={ess_min:.0f}/{N_m_p}")

        # Refresh the pilot-chain: Reset the IS origin if ESS collapsed for any of the three gain
        # evaluations, then re-evaluate all three gains fresh against the refreshed pilot chain.
        if ess_min < ess_refresh_frac * N_m_p:
            a0, b0 = dg_state["a"], dg_state["b"]
            F_hat0, rho_hat0 = F_hat, rho_hat
            tail = pilot_samples[-10:] if len(pilot_samples) else []
            candidates = list(tail[::-1]) + [dg_state["q_hd"]]
            log_pi_0, pilot_samples, pilot_acc = _draw_pilot(a0, b0, F_hat0, rho_hat0, candidates)
            diagnostics["pilot_refreshes"].append(dict(it=it, ess_before=ess_min, acc_rate=pilot_acc))
            if verbose:
                print(f"    [pilot refresh at it={it}] ess_min was "
                      f"{ess_min:.1f}/{N_m_p}; redrew pilot chain (acc_rate={pilot_acc:.3f})")
            G_DG, ess_dg = _gain("DG", dg_before, dg_after, C_DG_last, dg_state_before, dg_state)
            G_CDF, ess_cdf = _gain("CDF", F_hat_before, F_hat, C_CDF_last, state_CDF_before, state_CDF)
            G_PDF, ess_pdf = _gain("PDF", rho_hat_before, rho_hat, C_PDF_last, state_PDF_before, state_PDF)

    # Spend the remaining budget on DG as it is cheapest and has no overshoot risk
    n_leftover = int(np.floor((C_res - C_DG - C_CDF - C_PDF) / c_dg))
    if n_leftover > 0:
        dg_state = resample_delta_gamma(dg_state, n_leftover, dg_sampler, alpha, c_dg=c_dg_unit)
        C_DG = dg_state["C_DG"]

    # Compute the production MH chain from the final pi_hat, and build final estimate
    log_pi_final = build_log_pi_hat_from_callables(
        dg_state["a"], dg_state["b"], F_hat, rho_hat, S_0, S_1)
    prod_candidates = ([dg_state["q_hd"]]
                        + (list(pilot_samples[-10:][::-1]) if len(pilot_samples) else []))
    prod_start = _find_valid_start(log_pi_final, prod_candidates)
    if tune_s_mh:
        current_s_mh, prod_tuned_ar = tune_step_size(
            log_pi_final, prod_start, initial_guess=current_s_mh,
            target_ar=s_mh_target_ar, ar_tol=s_mh_ar_tol,
            pilot_n=s_mh_pilot_n, rng=rng)
    prod_samples, prod_acc = grw_metropolis_hastings(
        log_pi_final, prod_start, current_s_mh, N_m,
        burn_in=int(mh_burn_in_frac * N_m), rng=rng)

    correction = np.mean(
        prod_samples - dg_state["F_inv_dg"](F_hat(prod_samples)))
    q_hat = dg_state["q_hd"] + correction

    diagnostics.update(dict(
        n_iterations=it, N_dg_final=dg_state["N_dg"],
        C_DG=C_DG, C_CDF=C_CDF, C_PDF=C_PDF,
        C_total=C_DG + C_CDF + C_PDF + C_MH,
        production_mh_acc_rate=prod_acc, s_mh_final=current_s_mh,
        q_HD=dg_state["q_hd"], correction=correction,
        n_pilot_refreshes=len(diagnostics["pilot_refreshes"]),
        # Final GNR MLMC interpolants the estimator ends up using, over [S_0, S_1], plus the beta
        # kernel params (a, b) so the tilted target density g(F_hat; a, b) * rho_hat can be rebuilt:
        F_hat=F_hat, rho_hat=rho_hat, S_0=S_0, S_1=S_1, q_hat=q_hat,
        a=dg_state["a"], b=dg_state["b"],
        # Order-statistic quantile of the DG samples:
        dg_q_os=order_statistic_quantile(dg_state["sorted_samples"], alpha),
        # Production MH chain states (after burn-in / thinning) the final estimate is built from:
        prod_samples=prod_samples,
        # Final GNR smoothing/interpolation parameters actually used (delta_m, k_n, r) for the
        # MLMC CDF and PDF:
        cdf_delta=state_CDF["delta_m"], cdf_k=state_CDF["k_n"], cdf_r=state_CDF["r"],
        pdf_delta=state_PDF["delta_m"], pdf_k=state_PDF["k_n"], pdf_r=state_PDF["r"],
    ))

    if verbose:
        print(f"\nFinal: N_dg={dg_state['N_dg']}  C_DG={C_DG:.0f}  "
              f"C_CDF={C_CDF:.0f}  C_PDF={C_PDF:.0f}  "
              f"pilot_refreshes={len(diagnostics['pilot_refreshes'])}  "
              f"q_HD={dg_state['q_hd']:.4f}  correction={correction:.4f}  "
              f"q_hat={q_hat:.4f}")

    return q_hat, diagnostics


# Delta-gamma coarse-model bootstrap state (from dg_estimators): the running HD quantile,
# empirical quantile function, and resample procedure over accumulated DG samples.

def build_empirical_qf(sorted_samples: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Empirical quantile function via linear interpolation on the  empirical CDF.

    Returns a vectorized callable qf(u) -> value(s), u in [0, 1].
    """
    n = len(sorted_samples)
    p_grid = (np.arange(1, n + 1) - 0.5) / n

    def qf(u: np.ndarray) -> np.ndarray:
        """Evaluate the empirical quantile function at u in [0, 1]."""
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        return np.interp(u, p_grid, sorted_samples)

    return qf


def init_delta_gamma(dg_sampler: Callable[[int], np.ndarray], n_init: int, alpha: float,
                     c_dg: float = 1.0) -> dict:
    """
    Draw the initial n_init DG samples and build a fresh dg_state.
    This is equivalent to calling resample_delta_gamma(None, n_init, ...).
    """

    return resample_delta_gamma(None, n_init, dg_sampler, alpha, c_dg=c_dg)


def resample_delta_gamma(dg_state: dict | None, n_new: int, dg_sampler: Callable[[int], np.ndarray],
                         alpha: float, c_dg: float = 1.0) -> dict:
    """
    Draw n_new additional i.i.d. DG samples and update everything that
    depends on them: N_dg, a, b, q_hd (Harrell-Davis), F_inv_dg
    (empirical QF), and the cumulative DG cost C_DG.
    """
    if dg_state is None:
        prev_samples = np.empty(0, dtype=float)
        prev_C_DG = 0.0
    else:
        prev_samples = dg_state["samples"]
        prev_C_DG = dg_state["C_DG"]

    if n_new > 0:
        new_samples = dg_sampler(n_new)
        samples = np.concatenate([prev_samples, new_samples])
    else:
        samples = prev_samples

    N_dg = len(samples)
    a = alpha * (N_dg + 1)
    b = (1 - alpha) * (N_dg + 1)
    sorted_samples = np.sort(samples)
    q_hd = harrell_davis(sorted_samples, alpha)
    F_inv_dg = build_empirical_qf(sorted_samples)
    C_DG = prev_C_DG + n_new * c_dg

    return dict(
        samples=samples,
        sorted_samples=sorted_samples,
        N_dg=N_dg,
        a=a, b=b,
        q_hd=q_hd,
        F_inv_dg=F_inv_dg,
        C_DG=C_DG,
    )


# Adaptive cost-targeting step (from adapt_step_cost): a chain of warm-started, tightening
# eps probes until the cumulative warm_fn cost lands near C_target.

def _propose_next_eps(history: list[tuple[float, float]], C_target: float, eps_achieved: float,
                      blind_shrink: float = 0.9, shrink_min: float = 0.85,
                      shrink_max: float = 0.95) -> float:
    """
    Propose the next (strictly smaller) eps to try for MLMC CDF/PDF.

    This uses the model
        incremental_cost(eps) = K * (eps^-2 - eps_achieved^-2)
    where eps_achieved is the known starting tolerance for this entire
    adapt_step_cost call. K is fit via one-parameter least squares from
    whatever trials have been observed so far in history.
    """
    eps_current = history[-1][0]
    baseline = 0.0 if not np.isfinite(eps_achieved) else eps_achieved ** -2
    nonzero = [(e, c) for e, c in history if c > 0]

    if len(nonzero) >= 1:
        x = np.array([e ** -2 - baseline for e, _ in nonzero])
        y = np.array([c for _, c in nonzero])
        denom = np.sum(x * x)
        if denom > 0:
            # 1-parameter least squares through the origin
            K_est = np.sum(x * y) / denom
            if K_est > 0:
                target_x = C_target / K_est + baseline
                if target_x > 0:
                    eps_next = target_x ** -0.5
                    return float(np.clip(eps_next,
                                          shrink_min * eps_current,
                                          shrink_max * eps_current))

    return eps_current * blind_shrink


def adapt_step_cost(state: dict | None, C_target: float, S_0: float, S_1: float, r: int,
                    cost_M: float, lvl0: Callable, lvl1: Callable, warm_fn: Callable,
                    eps_hint: float = 0.1, cost_tol: float = 0.15, max_probes: int = 30,
                    safe_loose_eps: float = 1.0) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Find targ_eps such that warm_fn's cumulative cost across a chain of
    strictly-tightening, warm-started trials (starting from "state")
    lands close to C_target, and return the final chained state and the
    true cumulative cost of reaching it.
    """

    eps_achieved = state["eps_achieved"] if state is not None else np.inf
    is_free_anchor = not np.isfinite(eps_achieved)

    current_state = state
    cumulative_cost = 0.0
    n_probes_used = 0
    history = []
    s, v = None, None

    def probe(eps: float) -> None:
        """Run one warm-started warm_fn call at eps and fold its cost into the running total."""
        nonlocal current_state, cumulative_cost, n_probes_used, s, v
        was_fresh = current_state is None
        s, v, st, ic = warm_fn(eps, S_0, S_1, r, cost_M, lvl0, lvl1,
                                state=current_state)
        cost = ic["cost_total"]
        if was_fresh:
            cost += initial_state_cost(st)["cost_total"]
        cumulative_cost += cost
        current_state = st
        n_probes_used += 1
        history.append((eps, cumulative_cost))

    def result() -> tuple[np.ndarray, np.ndarray, dict, dict]:
        """Return the current (s_grid, vals, state, cumulative-cost dict)."""
        return s, v, current_state, dict(cost_total=cumulative_cost)

    def within_tol() -> bool:
        """Whether the cumulative cost is within cost_tol of C_target."""
        return abs(cumulative_cost - C_target) <= cost_tol * C_target

    # Phase 0: establish a working starting point
    if is_free_anchor:
        probe(safe_loose_eps)
        eps_achieved_for_model = np.inf
    else:
        probe(eps_achieved)
        eps_achieved_for_model = eps_achieved

    if within_tol() or cumulative_cost >= C_target:
        return result()

    # Phase 1: adaptive monotonic tightening
    while n_probes_used < max_probes:
        eps_next = _propose_next_eps(history, C_target, eps_achieved_for_model)
        probe(eps_next)
        if within_tol() or cumulative_cost >= C_target:
            return result()

    return result()

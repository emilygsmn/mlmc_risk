"""
Warm-startable adjustment of GNR's adaptive MLMC CDF and PDF estimators.

Consists of two core functions:

    warm_adaptive_mlmc (targ_eps, S_0, S_1, r, cost_M, lvl0, lvl1, state=None)
    warm_adaptive_mlmc_pdf (targ_eps, S_0, S_1, r, cost_M, lvl0, lvl1, state=None)

Both return (s_grid, vals, new_state, incr_cost).
"""

from collections.abc import Callable

import numpy as np

from mlmc_risk_estimation.blhd.mlmc_cdf import (
    apply_smoothing_with_g,
    estimate_expectations_of_g,
    compute_variance_estimates,
    compute_c_kn,
    smoothing_error,
    interpolation_error,
    compute_optimal_n,
)

from mlmc_risk_estimation.blhd.mlmc_pdf import (
    apply_smoothing_pdf,
    estimate_expectations_pdf,
    compute_variance_estimates_pdf,
    smoothing_error_pdf,
    interpolation_error_pdf,
)

__all__ = ["warm_adaptive_mlmc", "warm_adaptive_mlmc_pdf", "initial_state_cost"]

# utilities

def _kn(n: int, r: int) -> int:
    """GNR's knot-counting scheme k_n = ceil(2^n / r) * r + 1."""
    return int(np.ceil(2 ** n / r) * r + 1)

def _n_min(r: int) -> int:
    """Smallest n satisfying GNR 2017's stated precondition for k_n = ceil(2^n/r)*r+1
    to be well-defined and meaningfully increasing.
    """
    n = 1
    while 2 ** (n + 1) <= r:
        n += 1
    return n

def _delta(m: int, S_0: float, S_1: float) -> float:
    """Smoothing width at level m: (S_1 - S_0) / 2^m."""
    return (S_1 - S_0) / 2 ** m

def _zero_cost() -> dict:
    """Zero-cost incr_cost dict.."""
    return dict(delta_N_0=0, delta_N_1=0, did_resmooth=False,
                cost_dg=0.0, cost_f=0.0, cost_total=0.0)

def _make_incr_cost(delta_N_0: int, delta_N_1: int, cost_M: float, did_resmooth: bool) -> dict:
    """Build the incr_cost dict for a call that drew delta_N_0/delta_N_1 new samples."""

    return dict(
        delta_N_0=int(delta_N_0),
        delta_N_1=int(delta_N_1),
        did_resmooth=bool(did_resmooth),
        cost_dg=float(delta_N_0),
        cost_f=float(delta_N_1) * float(cost_M),
        cost_total=float(delta_N_0) + float(delta_N_1) * float(cost_M),
    )

def _check_compat(state: dict, S_0: float, S_1: float, r: int, cost_M: float) -> None:
    """Raise ValueError if `state` was not built with the same (S_0, S_1, r, cost_M)."""
    if state["S_0"] != S_0 or state["S_1"] != S_1:
        raise ValueError(
            f"State (S_0,S_1)=({state['S_0']},{state['S_1']}) "
            f"!= call ({S_0},{S_1})")
    if state["r"] != r:
        raise ValueError(f"State r={state['r']} != call r={r}")
    if abs(state["cost_M"] - cost_M) > 1e-12:
        raise ValueError(
            f"State cost_M={state['cost_M']} != call cost_M={cost_M}")

# The inner variance loop
# should be callable from all three places without duplication:
#  check 1 (warm restart, variance violated at current k_n/delta_m)
#  check 2 body (smoothing loop, each iteration)
#  check 3 body (interpolation loop, each m-step inside)

def _run_variance_loop(
        N_0: int, N_1: int,
        sm_c0: np.ndarray, sm_f: np.ndarray, sm_c1: np.ndarray,
        raw_c0: np.ndarray, raw_f: np.ndarray, raw_c1: np.ndarray,
        b_hat_0: np.ndarray, b_hat_1: np.ndarray, v_hat_0: float, v_hat_1: float,
        k_n: int, delta_m: float, c_kn: float, cost_M: float, eps_star: float,
        lvl0: Callable, lvl1: Callable, S_0: float, S_1: float, r: int,
        smoothing_fn: Callable, expectations_fn: Callable, variance_fn: Callable
        ) -> tuple:
    """GNR's innermost repeat-until variance loop at fixed (k_n, delta_m).

    New samples are smoothed incrementally because (k_n, delta_m) is fixed.
    """
    variance_targ_error = 256 * eps_star ** 2
    delta_N_0 = 0
    delta_N_1 = 0

    while True:
        n_0_opt, n_1_opt = compute_optimal_n(
            v_hat_0, v_hat_1, cost_M, k_n, delta_m, c_kn, eps_star)
        d0 = max(n_0_opt - N_0, 0)
        d1 = max(n_1_opt - N_1, 0)

        if d0 > 0:
            c0, _ = lvl0(d0)
            raw_c0 = np.concatenate([raw_c0, c0])
            sm_c0  = np.concatenate(
                [sm_c0, smoothing_fn(c0, S_0, S_1, k_n, delta_m, r)], axis=0)
            N_0 += d0;  delta_N_0 += d0

        if d1 > 0:
            f, c1, _ = lvl1(d1)
            raw_f  = np.concatenate([raw_f,  f])
            raw_c1 = np.concatenate([raw_c1, c1])
            sm_f   = np.concatenate(
                [sm_f,  smoothing_fn(f,  S_0, S_1, k_n, delta_m, r)], axis=0)
            sm_c1  = np.concatenate(
                [sm_c1, smoothing_fn(c1, S_0, S_1, k_n, delta_m, r)], axis=0)
            N_1 += d1;  delta_N_1 += d1

        b_hat_0, b_hat_1 = expectations_fn(sm_f, sm_c1, sm_c0)
        v_hat_0, v_hat_1 = variance_fn(
            sm_c0, sm_f, sm_c1, b_hat_0, b_hat_1,
            N_0, N_1, cost_M, k_n, delta_m)

        v_hat = c_kn * (v_hat_0 / N_0 + v_hat_1 / N_1)
        if v_hat <= variance_targ_error:
            break
        if d0 == 0 and d1 == 0:
            break

    return (N_0, N_1, sm_c0, sm_f, sm_c1, raw_c0, raw_f, raw_c1,
            b_hat_0, b_hat_1, v_hat_0, v_hat_1, delta_N_0, delta_N_1)

# Core warm-startable loop:

def _warm_loop(
    targ_eps: float, S_0: float, S_1: float, r: int, cost_M: float,
    lvl0: Callable, lvl1: Callable, state: dict,
    smoothing_fn: Callable, expectations_fn: Callable, variance_fn: Callable,
    smoothing_err_fn: Callable, interp_err_fn: Callable,
    clip_low: float | None, clip_high: float | None,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Run GNR's warm-startable nested variance/smoothing/interpolation loop from `state`
    to the given `targ_eps`."""

    # Constants
    C_r = 2 ** (r + 1)
    NQr = 1.63

    BIAS_FREE_EPS_STAR_COEF = 5 + 16 * np.sqrt(2)

    eps_star          = targ_eps / (BIAS_FREE_EPS_STAR_COEF * NQr)
    interp_targ_error = NQr * (C_r - 1) * eps_star
    smooth_targ_error = 4   * (C_r - 1) * eps_star

    n       = state["n"];       m       = state["m"]
    k_n     = state["k_n"];     delta_m = state["delta_m"]
    c_kn    = state["c_kn"]
    N_0     = state["N_0"];     N_1     = state["N_1"]
    raw_c0  = state["raw_c0"].copy()
    raw_f   = state["raw_f"].copy()
    raw_c1  = state["raw_c1"].copy()
    sm_c0   = state["sm_c0"];   sm_f  = state["sm_f"];   sm_c1  = state["sm_c1"]
    b_hat_0 = state["b_hat_0"]; b_hat_1 = state["b_hat_1"]
    v_hat_0 = state["v_hat_0"]; v_hat_1 = state["v_hat_1"]
    i_hat   = state["i_hat"];   s_hat  = state["s_hat"]

    init_N_0     = N_0
    init_N_1     = N_1
    did_resmooth = False

    def _var() -> None:
        """Run the inner variance loop at the current (k_n, delta_m)."""

        nonlocal N_0, N_1, sm_c0, sm_f, sm_c1
        nonlocal raw_c0, raw_f, raw_c1, b_hat_0, b_hat_1, v_hat_0, v_hat_1
        (N_0, N_1, sm_c0, sm_f, sm_c1, raw_c0, raw_f, raw_c1,
         b_hat_0, b_hat_1, v_hat_0, v_hat_1, _, _) = _run_variance_loop(
            N_0, N_1, sm_c0, sm_f, sm_c1, raw_c0, raw_f, raw_c1,
            b_hat_0, b_hat_1, v_hat_0, v_hat_1,
            k_n, delta_m, c_kn, cost_M, eps_star,
            lvl0, lvl1, S_0, S_1, r,
            smoothing_fn, expectations_fn, variance_fn)

    def _resmooth() -> None:
        """Re-smooth all raw samples at the current (k_n, delta_m).
        Called whenever k_n or delta_m changes (i.e. on every m- or n-step)."""
        nonlocal sm_c0, sm_f, sm_c1, b_hat_0, b_hat_1, v_hat_0, v_hat_1
        nonlocal did_resmooth
        sm_c0 = smoothing_fn(raw_c0, S_0, S_1, k_n, delta_m, r)
        sm_f  = smoothing_fn(raw_f,  S_0, S_1, k_n, delta_m, r)
        sm_c1 = smoothing_fn(raw_c1, S_0, S_1, k_n, delta_m, r)
        b_hat_0, b_hat_1 = expectations_fn(sm_f, sm_c1, sm_c0)
        v_hat_0, v_hat_1 = variance_fn(
            sm_c0, sm_f, sm_c1, b_hat_0, b_hat_1,
            N_0, N_1, cost_M, k_n, delta_m)
        did_resmooth = True

    # Warm-restart entry-point selection. This is skipped entirely for fresh starts

    if i_hat < np.inf:

        # Check 1: variance condition
        v_hat = c_kn * (v_hat_0 / N_0 + v_hat_1 / N_1)
        if v_hat > 256 * eps_star ** 2:
            _var()
            # n >= 2, m >= 2 guaranteed for any state returned by a prior call.
            s_hat = smoothing_err_fn(
                raw_f, S_0, S_1, k_n, delta_m, _delta(m - 1, S_0, S_1), r)
            i_hat = interp_err_fn(
                raw_f, S_0, S_1, k_n, _kn(n - 1, r), delta_m, r)

        # Check 2: smoothing, with variance nested inside.
        if s_hat > smooth_targ_error:
            while s_hat > smooth_targ_error:
                m      += 1
                delta_m      = _delta(m,   S_0, S_1)
                delta_m_prev = _delta(m-1, S_0, S_1)
                _resmooth() # full re-smooth at new (k_n, delta_m)
                _var() # variance nested inside smoothing
                s_hat = smoothing_err_fn(
                    raw_f, S_0, S_1, k_n, delta_m, delta_m_prev, r)
            i_hat = interp_err_fn(
                raw_f, S_0, S_1, k_n, _kn(n - 1, r), delta_m, r)

    # Check 3: standard GNR outer interpolation loop.

    while i_hat > interp_targ_error:
        n   += 1
        m   -= 1
        k_n  = _kn(n, r)
        c_kn = compute_c_kn(k_n)
        s_hat = np.inf # force smoothing loop at new k_n

        while s_hat > smooth_targ_error:
            m      += 1
            delta_m      = _delta(m,   S_0, S_1)
            delta_m_prev = _delta(m-1, S_0, S_1)
            _resmooth() # full re-smooth at new (k_n, delta_m)
            _var() # variance nested inside smoothing
            s_hat = smoothing_err_fn(
                raw_f, S_0, S_1, k_n, delta_m, delta_m_prev, r)

        i_hat = interp_err_fn(
            raw_f, S_0, S_1, k_n, _kn(n - 1, r), delta_m, r)

    # Build the output
    s    = np.linspace(S_0, S_1, len(b_hat_0))
    vals = b_hat_0 + b_hat_1
    if clip_low  is not None: vals = np.clip(vals, clip_low,  None)
    if clip_high is not None: vals = np.clip(vals, None, clip_high)

    new_state = dict(
        n=n, m=m, k_n=k_n, delta_m=delta_m, c_kn=c_kn,
        N_0=N_0, N_1=N_1,
        raw_c0=raw_c0, raw_f=raw_f, raw_c1=raw_c1,
        sm_c0=sm_c0, sm_f=sm_f, sm_c1=sm_c1,
        b_hat_0=b_hat_0, b_hat_1=b_hat_1,
        v_hat_0=v_hat_0, v_hat_1=v_hat_1,
        i_hat=i_hat, s_hat=s_hat,
        S_0=S_0, S_1=S_1, r=r, cost_M=cost_M,
        eps_achieved=targ_eps,
    )
    return s, vals, new_state, _make_incr_cost(
        N_0 - init_N_0, N_1 - init_N_1, cost_M, did_resmooth)


# Create the fresh states

def _fresh_state(S_0: float, S_1: float, r: int, cost_M: float,
                  lvl0: Callable, lvl1: Callable,
                  smoothing_fn: Callable, expectations_fn: Callable, variance_fn: Callable,
                  N_init: int = 100) -> dict:
    """Build the initial warm-startable state from N_init+N_init seed samples (i_hat=s_hat=inf
    so the first _warm_loop call always runs the full outer interpolation loop)."""

    n, m    = _n_min(r), 2
    k_n     = _kn(n, r)
    delta_m = _delta(m, S_0, S_1)
    c_kn    = compute_c_kn(k_n)

    raw_c0, _        = lvl0(N_init)
    raw_f, raw_c1, _ = lvl1(N_init)

    sm_c0 = smoothing_fn(raw_c0, S_0, S_1, k_n, delta_m, r)
    sm_f  = smoothing_fn(raw_f,  S_0, S_1, k_n, delta_m, r)
    sm_c1 = smoothing_fn(raw_c1, S_0, S_1, k_n, delta_m, r)

    b_hat_0, b_hat_1 = expectations_fn(sm_f, sm_c1, sm_c0)
    v_hat_0, v_hat_1 = variance_fn(
        sm_c0, sm_f, sm_c1, b_hat_0, b_hat_1,
        N_init, N_init, cost_M, k_n, delta_m)

    return dict(
        n=n, m=m, k_n=k_n, delta_m=delta_m, c_kn=c_kn,
        N_0=N_init, N_1=N_init,
        raw_c0=raw_c0, raw_f=raw_f, raw_c1=raw_c1,
        sm_c0=sm_c0, sm_f=sm_f, sm_c1=sm_c1,
        b_hat_0=b_hat_0, b_hat_1=b_hat_1,
        v_hat_0=v_hat_0, v_hat_1=v_hat_1,
        i_hat=np.inf, s_hat=np.inf,    # inf forces outer loop on first call
        S_0=S_0, S_1=S_1, r=r, cost_M=cost_M,
        eps_achieved=np.inf,
    )



def warm_adaptive_mlmc(targ_eps: float, S_0: float, S_1: float, r: int, cost_M: float,
                       lvl0: Callable, lvl1: Callable, state: dict | None = None
                       ) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Warm-startable adaptive MLMC CDF estimator (GNR 2015/2017)."""

    if state is None:
        state = _fresh_state(S_0, S_1, r, cost_M, lvl0, lvl1,
                              apply_smoothing_with_g,
                              estimate_expectations_of_g,
                              compute_variance_estimates)
    else:
        _check_compat(state, S_0, S_1, r, cost_M)
        if targ_eps >= state.get("eps_achieved", np.inf):
            s = np.linspace(S_0, S_1, len(state["b_hat_0"]))
            return (s,
                    np.clip(state["b_hat_0"] + state["b_hat_1"], 0., 1.),
                    state, _zero_cost())

    return _warm_loop(
        targ_eps, S_0, S_1, r, cost_M, lvl0, lvl1, state,
        smoothing_fn    = apply_smoothing_with_g,
        expectations_fn = estimate_expectations_of_g,
        variance_fn     = compute_variance_estimates,
        smoothing_err_fn= smoothing_error,
        interp_err_fn   = interpolation_error,
        clip_low=0.0, clip_high=1.0,
    )


def warm_adaptive_mlmc_pdf(targ_eps: float, S_0: float, S_1: float, r: int, cost_M: float,
                           lvl0: Callable, lvl1: Callable, state: dict | None = None
                           ) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Warm-startable adaptive MLMC PDF estimator (analogous to GNR 2015/2017)."""

    if state is None:
        state = _fresh_state(S_0, S_1, r, cost_M, lvl0, lvl1,
                              apply_smoothing_pdf,
                              estimate_expectations_pdf,
                              compute_variance_estimates_pdf)
    else:
        _check_compat(state, S_0, S_1, r, cost_M)
        if targ_eps >= state.get("eps_achieved", np.inf):
            s = np.linspace(S_0, S_1, len(state["b_hat_0"]))
            return (s,
                    np.clip(state["b_hat_0"] + state["b_hat_1"], 0., None),
                    state, _zero_cost())

    return _warm_loop(
        targ_eps, S_0, S_1, r, cost_M, lvl0, lvl1, state,
        smoothing_fn    = apply_smoothing_pdf,
        expectations_fn = estimate_expectations_pdf,
        variance_fn     = compute_variance_estimates_pdf,
        smoothing_err_fn= smoothing_error_pdf,
        interp_err_fn   = interpolation_error_pdf,
        clip_low=0.0, clip_high=None,
    )


def initial_state_cost(state: dict) -> dict:
    """Charge the initial costs."""

    return _make_incr_cost(state["N_0"], state["N_1"], state["cost_M"], False)

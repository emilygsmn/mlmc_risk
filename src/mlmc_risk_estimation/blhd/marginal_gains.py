"""
marginal_gains.py

Evaluates the marginal accuracy gain each of DG / CDF / PDF's OWN LAST
ACTUAL investment delivered, per unit of that investment's real cost --
reweighted, via importance sampling, under whatever the CURRENT shared
context happens to be right now.

For each of DG/CDF/PDF, a "before" and "after" snapshot of just that
component's own axis, is saved, taken at the time of its own last actual
investment.

Evaluating a gain never requires new MLMC/DG runs for a component that
is not being invested in in this iteration. It only requires reweighting
the pilot sample under these existing, snapshots.

The only real compute per iteration is generating a new "after" snapshot
for whichever component just won the race.

Concretely:
    DG  influences  (a, b, F_inv_dg, q_hd)
    CDF influences  F_hat
    PDF influences  rho_hat

For whichever axis is not varying, both the before and after evaluation
use the current shared values of that axis.

The pilot chain {L_tilde_i} is drawn once, from the density pi_hat_0
implied by the state at that moment (a0, b0, F_hat0, rho_hat0), and
reused, unchanged, for the entire race loop. Unless ESS triggers a restart.

Self-normalized importance sampling requires the reference density in the
weight ratio to be the density the pilot samples were actually drawn from.
So, we reweight from this same fixed pi_hat_0.
"""

from collections.abc import Callable

import numpy as np
from mlmc_risk_estimation.blhd.mh import build_log_pi_hat_from_callables

__all__ = ["evaluate_gain"]


def _self_normalized_is_correction(pilot_samples: np.ndarray,
                                    log_pi_new: Callable[[np.ndarray], np.ndarray],
                                    log_pi_0: Callable[[np.ndarray], np.ndarray],
                                    F_hat_corr: Callable[[np.ndarray], np.ndarray],
                                    F_inv_dg_corr: Callable[[np.ndarray], np.ndarray]
                                    ) -> tuple[float, float]:
    """Self-normalized importance-sampling estimate of
    E_{pi_new}[ L - F_inv_dg_corr(F_hat_corr(L)) ], using samples L ~ pi_0 (the fixed
    pilot-draw-time density) with importance weights.

    ESS is the effective sample size (diagnostic for weight degeneracy as pi_new drifts
    far from pi_0).
    """
    
    log_w = log_pi_new(pilot_samples) - log_pi_0(pilot_samples)
    finite = np.isfinite(log_w)
    if not np.any(finite):
        return np.nan, 0.0
    log_w = log_w - np.max(log_w[finite])
    w = np.where(finite, np.exp(log_w), 0.0)

    corr = pilot_samples - F_inv_dg_corr(F_hat_corr(pilot_samples))
    sum_w = np.sum(w)
    if sum_w <= 0:
        return np.nan, 0.0
    Delta = np.sum(w * corr) / sum_w
    ess = (sum_w ** 2) / np.sum(w ** 2)
    return float(Delta), float(ess)


def evaluate_gain(pilot_samples: np.ndarray,
                   log_pi_0: Callable[[np.ndarray], np.ndarray],
                   S_0: float, S_1: float,
                   axis: str,
                   before: tuple | Callable[[np.ndarray], np.ndarray],
                   after: tuple | Callable[[np.ndarray], np.ndarray],
                   cost_last: float,
                   a_shared: float, b_shared: float,
                   F_hat_shared: Callable[[np.ndarray], np.ndarray],
                   rho_hat_shared: Callable[[np.ndarray], np.ndarray],
                   F_inv_dg_shared: Callable[[np.ndarray], np.ndarray],
                   q_hd_shared: float) -> tuple[float, float, float]:
    """Evaluate the marginal gain per unit cost that one component's own last actual
    investment delivered, reweighted under the current shared context.
    """

    if axis == "DG":
        a_b, b_b, Finv_b, qhd_b = before
        a_a, b_a, Finv_a, qhd_a = after
        log_pi_b = build_log_pi_hat_from_callables(a_b, b_b, F_hat_shared, rho_hat_shared, S_0, S_1)
        log_pi_a = build_log_pi_hat_from_callables(a_a, b_a, F_hat_shared, rho_hat_shared, S_0, S_1)
        Delta_b, ess_b = _self_normalized_is_correction(pilot_samples, log_pi_b, log_pi_0, F_hat_shared, Finv_b)
        Delta_a, ess_a = _self_normalized_is_correction(pilot_samples, log_pi_a, log_pi_0, F_hat_shared, Finv_a)
        q_b = qhd_b + Delta_b
        q_a = qhd_a + Delta_a
        gain = abs(q_a - q_b)

    elif axis == "CDF":
        F_hat_b, F_hat_a = before, after
        log_pi_b = build_log_pi_hat_from_callables(a_shared, b_shared, F_hat_b, rho_hat_shared, S_0, S_1)
        log_pi_a = build_log_pi_hat_from_callables(a_shared, b_shared, F_hat_a, rho_hat_shared, S_0, S_1)
        Delta_b, ess_b = _self_normalized_is_correction(pilot_samples, log_pi_b, log_pi_0, F_hat_b, F_inv_dg_shared)
        Delta_a, ess_a = _self_normalized_is_correction(pilot_samples, log_pi_a, log_pi_0, F_hat_a, F_inv_dg_shared)
        gain = abs(Delta_a - Delta_b)

    elif axis == "PDF":
        rho_b, rho_a = before, after
        log_pi_b = build_log_pi_hat_from_callables(a_shared, b_shared, F_hat_shared, rho_b, S_0, S_1)
        log_pi_a = build_log_pi_hat_from_callables(a_shared, b_shared, F_hat_shared, rho_a, S_0, S_1)
        Delta_b, ess_b = _self_normalized_is_correction(pilot_samples, log_pi_b, log_pi_0, F_hat_shared, F_inv_dg_shared)
        Delta_a, ess_a = _self_normalized_is_correction(pilot_samples, log_pi_a, log_pi_0, F_hat_shared, F_inv_dg_shared)
        gain = abs(Delta_a - Delta_b)

    else:
        raise ValueError(f"axis must be 'DG', 'CDF', or 'PDF'; got {axis!r}")

    if cost_last <= 0:
        return 0.0, ess_b, ess_a
    return gain / cost_last, ess_b, ess_a

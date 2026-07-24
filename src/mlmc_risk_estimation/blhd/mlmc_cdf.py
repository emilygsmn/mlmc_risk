"""GNR adaptive-MLMC helper functions for CDF estimation."""

import numpy as np

from mlmc_risk_estimation.blhd.gnr import gnr_cdf_values

__all__ = [
    "apply_smoothing_with_g",
    "estimate_expectations_of_g",
    "compute_c_kn",
    "smoothing_error",
    "interpolation_error",
    "compute_variance_estimates",
    "compute_optimal_n",
]

def _g_for_cdf(t: np.ndarray, r: int) -> np.ndarray:
    """Degree-r smoothing kernel for the indicator 1_{X<=s} (GNR polynomial mollifier)."""
    t = np.asarray(t)
    g = np.zeros_like(t, dtype=float)
    g[t <= 1] = 1.0
    mask = (t > -1) & (t < 1)
    tm = t[mask]
    if r in (0, 1):
        g[mask] = 0.5 - 0.5 * tm
    elif r in (2, 3):
        g[mask] = 0.5 - 9 / 8 * tm + 5 / 8 * tm ** 3
    elif r in (4, 5):
        g[mask] = 0.5 - 225 / 128 * tm + 175 / 64 * tm ** 3 - 189 / 128 * tm ** 5
    else:
        raise ValueError("r must be in (0, 1, 2, 3, 4, 5)")
    return g

def _compute_matrix_for_cdf(samples: np.ndarray, s: np.ndarray, delta: float, r: int) -> np.ndarray:
    """Smoothed-indicator matrix g((samples[i] - s[j]) / delta) for every sample/knot pair."""

    samples = np.asarray(samples)
    s = np.asarray(s)
    t = (samples[:, None] - s[None, :]) / delta
    return _g_for_cdf(t, r)

def apply_smoothing_with_g(samples: np.ndarray, S_0: float, S_1: float, k: int,
                           delta: float, r: int) -> np.ndarray:
    """Smoothed-indicator matrix of `samples` against the k equidistant knots on [S_0, S_1]."""

    s = np.linspace(S_0, S_1, int(k))
    return _compute_matrix_for_cdf(samples, s, delta, r)

def estimate_expectations_of_g(g_mat_fine: np.ndarray, g_mat_coarse_lvl1: np.ndarray,
                               g_mat_coarse_lvl0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MLMC level-0 estimate and level-1 correction of E[g] at each knot."""

    lvl0 = g_mat_coarse_lvl0.mean(axis=0)
    corr = (g_mat_fine - g_mat_coarse_lvl1).mean(axis=0)
    return lvl0, corr

def compute_c_kn(k_n: int) -> float:
    """GNR's error constant c_{k_n} (shared by the CDF and PDF estimators)."""

    gamma_sq = np.log(k_n + 1) + np.sqrt(8 / np.pi) * sum(
        1.0 / (np.sqrt(np.log(j)) * j ** 2) for j in range(2, k_n + 2)
    )
    return np.sqrt(2.0 * np.pi) * np.sqrt(gamma_sq)

def smoothing_error(fine_samples: np.ndarray, S_0: float, S_1: float, k_n: int,
                    delta_m: float, delta_m_prev: float, r: int) -> float:
    """s_hat: sup-norm change in the smoothed estimate between successive smoothing widths."""

    mean_m = apply_smoothing_with_g(fine_samples, S_0, S_1, k_n, delta_m, r).mean(axis=0)
    mean_m_prev = apply_smoothing_with_g(fine_samples, S_0, S_1, k_n, delta_m_prev, r).mean(axis=0)
    return np.max(np.abs(mean_m - mean_m_prev))

def interpolation_error(fine_samples: np.ndarray, S_0: float, S_1: float, k_n: int,
                        k_n_prev: int, delta_m: float, r: int) -> float:
    """i_hat: sup-norm change in GNR's corrected CDF Q_k^r between successive knot counts."""

    mean_kn = apply_smoothing_with_g(fine_samples, S_0, S_1, k_n, delta_m, r).mean(axis=0)
    mean_kn_prev = apply_smoothing_with_g(fine_samples, S_0, S_1, k_n_prev, delta_m, r).mean(axis=0)
    x_n = np.linspace(0.0, 1.0, k_n)
    x_prev = np.linspace(0.0, 1.0, k_n_prev)
    x_fine = np.linspace(0.0, 1.0, max(k_n, k_n_prev) * r)
    qn_vals = gnr_cdf_values(x_n, mean_kn, x_fine, r)
    qn_prev_vals = gnr_cdf_values(x_prev, mean_kn_prev, x_fine, r)
    return np.max(np.abs(qn_vals - qn_prev_vals))

def _compute_n_prime(N_ell: int, cost_ell: float, k_n: int, zeta: float) -> int:
    """Cost-capped subsample size for the variance estimate across the mas of knots, at level ell."""

    return int(min(N_ell, max(zeta, N_ell * cost_ell / k_n)))

def compute_variance_estimates(sm_c0: np.ndarray, sm_fine: np.ndarray, sm_c1: np.ndarray,
                               b_hat_0: np.ndarray, b_hat_1: np.ndarray, N_0: int, N_1: int,
                               cost_M: float, k_n: int, delta_m: float) -> tuple[float, float]:
    """Level-0 and level-1 variance estimates (max over knots), on a cost-capped subsample."""

    c_total = N_0 * (1 + k_n * delta_m) + N_1 * (cost_M + k_n * delta_m)
    zeta = k_n + c_total
    N_0_prime = _compute_n_prime(N_0, 1, k_n, zeta)
    N_1_prime = _compute_n_prime(N_1, cost_M, k_n, zeta)
    v_hat_0 = np.mean(np.max(np.abs(sm_c0[:N_0_prime] - b_hat_0), axis=1) ** 2)
    v_hat_1 = np.mean(np.max(np.abs(sm_fine[:N_1_prime] - sm_c1[:N_1_prime] - b_hat_1), axis=1) ** 2)
    return v_hat_0, v_hat_1

def compute_optimal_n(v_hat_0: float, v_hat_1: float, cost_M: float, k_n: int, delta_m: float,
                      c_kn: float, eps_star: float) -> tuple[int, int]:
    """GNR's cost-optimal per-level sample counts for target sub-error eps_star."""
    
    cost_0 = 1 + k_n * delta_m
    cost_1 = cost_M + k_n * delta_m
    scale = np.sqrt(v_hat_0 * cost_0) + np.sqrt(v_hat_1 * cost_1)
    n_0 = int(np.ceil(np.sqrt(v_hat_0) / np.sqrt(cost_0) * scale * c_kn / (256 * eps_star ** 2)))
    n_1 = int(np.ceil(np.sqrt(v_hat_1) / np.sqrt(cost_1) * scale * c_kn / (256 * eps_star ** 2)))
    return n_0, n_1

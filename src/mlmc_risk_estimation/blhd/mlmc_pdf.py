"""GNR adaptive-MLMC helper functions for PDF (density) estimation."""

import numpy as np

from mlmc_risk_estimation.blhd.gnr import gnr_pdf_values
from mlmc_risk_estimation.blhd.mlmc_cdf import (
    estimate_expectations_of_g as estimate_expectations_pdf,
    compute_variance_estimates as compute_variance_estimates_pdf,
)

__all__ = [
    "apply_smoothing_pdf",
    "estimate_expectations_pdf",
    "compute_variance_estimates_pdf",
    "smoothing_error_pdf",
    "interpolation_error_pdf",
]

def _g_for_pdf(t: np.ndarray, r: int) -> np.ndarray:
    """Degree-r smoothing kernel for the density."""
    t = np.asarray(t)
    g = np.zeros_like(t, dtype=float)
    mask = (t > -1) & (t < 1)
    tm = t[mask]
    if r in (1, 2):
        g[mask] = 3 / 4 * (1 - tm ** 2)
    elif r in (3, 4):
        g[mask] = (45 - 150 * tm ** 2 + 105 * tm ** 4) / 32
    elif r in (5, 6):
        g[mask] = (525 - 5675 * tm ** 2 + 6615 * tm ** 4 - 3465 * tm ** 6) / 256
    else:
        raise ValueError("r must be in (1, 2, 3, 4, 5, 6)")
    return g

def apply_smoothing_pdf(samples: np.ndarray, S_0: float, S_1: float, k: int,
                        delta: float, r: int) -> np.ndarray:
    """Smoothed-density matrix of `samples` against the k equidistant knots on [S_0, S_1]."""

    s = np.linspace(S_0, S_1, int(k))
    t = (samples[:, None] - s[None, :]) / delta
    return _g_for_pdf(t, r) / delta

def smoothing_error_pdf(fine_samples: np.ndarray, S_0: float, S_1: float, k_n: int,
                        delta_m: float, delta_m_prev: float, r: int) -> float:
    """s_hat for the density: sup-norm change between successive smoothing widths."""

    mean_m = apply_smoothing_pdf(fine_samples, S_0, S_1, k_n, delta_m, r).mean(axis=0)
    mean_m_prev = apply_smoothing_pdf(fine_samples, S_0, S_1, k_n, delta_m_prev, r).mean(axis=0)
    return np.max(np.abs(mean_m - mean_m_prev))

def interpolation_error_pdf(fine_samples: np.ndarray, S_0: float, S_1: float, k_n: int,
                            k_n_prev: int, delta_m: float, r: int) -> float:
    """i_hat for the density: sup-norm change in GNR's degree-r density interpolant between
    successive knot counts.
    """

    mean_kn = apply_smoothing_pdf(fine_samples, S_0, S_1, k_n, delta_m, r).mean(axis=0)
    mean_kn_prev = apply_smoothing_pdf(fine_samples, S_0, S_1, k_n_prev, delta_m, r).mean(axis=0)
    x_n = np.linspace(0.0, 1.0, k_n)
    x_prev = np.linspace(0.0, 1.0, k_n_prev)
    x_fine = np.linspace(0.0, 1.0, max(k_n, k_n_prev) * r)
    qn_vals = gnr_pdf_values(x_n, mean_kn, x_fine, r)
    qn_prev_vals = gnr_pdf_values(x_prev, mean_kn_prev, x_fine, r)
    return np.max(np.abs(qn_vals - qn_prev_vals))

"""Post-processing of the MLMC CDF and PDF estimates (Giles, Nagapetyan, Ritter).

The raw MLMC output is a set of CDF/PDF values on an equidistant knot grid. This module
turns those into continuous, corrected callables following GNR's construction.

Interpolation (GNR 2015 / 2017): piecewise polynomial of degree r at
r+1 consecutive knots. This requires a grid of k = r*n + 1 knots.

CDF correction (GNR 2017 Remark 4), a two-stage map Q_k^r = (f + h) / 2:
    Stage 1, on the raw knot values y: project onto a non-decreasing sequence in [0, 1].
    Stage 2, on the piecewise interpolant phi of z: per knot cell running sup/inf.
  
    Each stage is Lipschitz-1, so Q_k^r keeps the plain interpolant's Lipschitz constant
    (~1.63 for r=3).

PDF correction (GNR 2015): 
    The pointwise floor s -> max(Q_k^r(s), 0).
    
    This enforces non-negativity without increasing the error.
"""

from collections.abc import Callable

import numpy as np
from scipy.interpolate import interp1d

__all__ = ["gnr_cdf_values", "gnr_pdf_values",
           "build_monotone_cdf_interp", "build_nonneg_pdf_interp"]

def _lagrange_eval(s_pts: np.ndarray, y_pts: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate the degree-(len(s_pts)-1) Lagrange polynomial through (s_pts, y_pts) at x.
    Linear in y_pts (required for the GNR Lipschitz-constant analysis to apply)."""

    x = np.asarray(x, dtype=float)
    d = len(s_pts)
    result = np.zeros_like(x)
    for j in range(d):
        basis = np.ones_like(x)
        for m in range(d):
            if m != j:
                basis = basis * (x - s_pts[m]) / (s_pts[j] - s_pts[m])
        result = result + y_pts[j] * basis
    return result

def _piecewise_poly_eval(s_grid: np.ndarray, y: np.ndarray, x_eval: np.ndarray,
                         degree: int) -> np.ndarray:
    """Piecewise degree-`degree` Lagrange interpolation on consecutive, endpoint-shared
    groups of degree+1 knots (GNR Remark 3). Requires len(s_grid) = degree*n + 1."""

    s_grid = np.asarray(s_grid, dtype=float)
    y = np.asarray(y, dtype=float)
    k = len(s_grid)
    d = max(int(degree), 1)
    if (k - 1) % d != 0:
        raise ValueError(f"Piecewise degree-{d} interpolation requires k = {d}*n+1 knots, got k={k}")
    n_seg = (k - 1) // d
    x_eval = np.asarray(x_eval, dtype=float)
    out = np.empty_like(x_eval)

    seg_edges = s_grid[0::d]
    seg_idx = np.clip(np.searchsorted(seg_edges, x_eval, side="right") - 1, 0, n_seg - 1)
    for seg in range(n_seg):
        mask = seg_idx == seg
        if not np.any(mask):
            continue
        i0 = d * seg
        out[mask] = _lagrange_eval(s_grid[i0:i0 + d + 1], y[i0:i0 + d + 1], x_eval[mask])
    return out

def _monotone_projection(y: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """GNR Remark 4: project knot values onto a non-decreasing sequence in
    [lo, hi] and average the forward/backward parts."""

    y = np.asarray(y, dtype=float)
    k = len(y)
    u = np.empty(k)
    running = lo
    for j in range(k):
        running = min(max(y[j], running), hi)
        u[j] = running
    v = np.empty(k)
    running = hi
    for j in range(k - 1, -1, -1):
        running = max(min(y[j], running), lo)
        v[j] = running
    return (u + v) / 2.0

def _local_supinf_correction(s_grid: np.ndarray, z: np.ndarray, x_eval: np.ndarray,
                             degree: int) -> np.ndarray:
    """GNR Remark 4: per-cell running sup/inf of the interpolant phi(z), returning
    (f + h) / 2. Assumes x_eval is sorted within each knot cell (true for the dense grid the
    CDF builder uses). The running accumulation would otherwise mis-order the sup/inf."""

    s_grid = np.asarray(s_grid, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    phi_at_knots = _piecewise_poly_eval(s_grid, z, s_grid, degree)
    phi_x = _piecewise_poly_eval(s_grid, z, x_eval, degree)

    k = len(s_grid)
    f_vals = np.empty_like(x_eval)
    h_vals = np.empty_like(x_eval)
    cell_idx = np.clip(np.searchsorted(s_grid, x_eval, side="right") - 1, 0, k - 2)
    for j in range(k - 1):
        mask = cell_idx == j
        if not np.any(mask):
            continue
        local = phi_x[mask]
        running_sup = np.maximum.accumulate(np.concatenate([[phi_at_knots[j]], local]))[1:]
        running_inf = np.minimum.accumulate(np.concatenate([local, [phi_at_knots[j + 1]]])[::-1])[::-1][:-1]
        f_vals[mask] = np.minimum(running_sup, phi_at_knots[j + 1])
        h_vals[mask] = np.maximum(running_inf, phi_at_knots[j])
    return (f_vals + h_vals) / 2.0

def gnr_cdf_values(s_grid: np.ndarray, y: np.ndarray, x_eval: np.ndarray, r: int) -> np.ndarray:
    """GNR's corrected CDF estimator Q_k^r = (f + h) / 2 evaluated at x_eval (two-stage
    monotone correction of the degree-r piecewise interpolant)."""

    z = _monotone_projection(np.asarray(y, dtype=float), lo=0.0, hi=1.0)
    return _local_supinf_correction(s_grid, z, x_eval, degree=r)

def gnr_pdf_values(s_grid: np.ndarray, y: np.ndarray, x_eval: np.ndarray, r: int) -> np.ndarray:
    """GNR's density interpolant: the raw degree-r piecewise-Lagrange interpolation (the
    non-negativity floor is applied separately by build_nonneg_pdf_interp)."""

    return _piecewise_poly_eval(s_grid, y, x_eval, degree=r)

def build_monotone_cdf_interp(s_grid: np.ndarray, y: np.ndarray, r: int,
                              oversample: int = 10) -> Callable[[np.ndarray], np.ndarray]:
    """Continuous, non-decreasing CDF callable for GNR's Q_k^r, built by densely sampling
    Q_k^r and linearly interpolating between them.
    """
    s_grid = np.asarray(s_grid, dtype=float)
    y = np.asarray(y, dtype=float)

    dense_x = np.linspace(s_grid[0], s_grid[-1], oversample * (len(s_grid) - 1) + 1)
    corrected = gnr_cdf_values(s_grid, y, dense_x, r)
    final_interp = interp1d(dense_x, corrected, kind="linear", bounds_error=False,
                            fill_value=(corrected[0], corrected[-1]))

    def evaluate(x: np.ndarray) -> np.ndarray:
        """Evaluate the corrected, continuous CDF at x."""
        return final_interp(x)

    evaluate._gnr_cdf_data = (np.ascontiguousarray(dense_x, dtype=np.float64),
                              np.ascontiguousarray(corrected, dtype=np.float64))
    return evaluate

def build_nonneg_pdf_interp(s_grid: np.ndarray, y: np.ndarray,
                            r: int) -> Callable[[np.ndarray], np.ndarray]:
    """Continuous, non-negative density callable: the degree-r piecewise-Lagrange interpolant
    with the pointwise floor max(., 0) and constant extrapolation outside the knot range.
    """

    s_grid = np.asarray(s_grid, dtype=float)
    y = np.asarray(y, dtype=float)

    def corrected(x: np.ndarray) -> np.ndarray:
        """Evaluate the corrected, non-negative density at x."""
        # Clip into the knot range (constant extrapolation)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        xc = np.clip(x_arr, s_grid[0], s_grid[-1])
        out = np.maximum(gnr_pdf_values(s_grid, y, xc, r), 0.0)
        return out if np.ndim(x) else out[0]

    corrected._gnr_pdf_data = (np.ascontiguousarray(s_grid, dtype=np.float64),
                               np.ascontiguousarray(y, dtype=np.float64), int(r))
    return corrected

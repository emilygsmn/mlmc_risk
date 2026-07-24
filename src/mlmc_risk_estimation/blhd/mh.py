"""Module providing a Gaussian random-walk Metropolis-Hastings sampler for the target density

    pi_hat(x) = g_{a,b}(F_hat(x)) * f_hat(x)

where g_{a,b} is the Beta(a,b) Harrell-Davis kernel, F_hat is the MLMC CDF estimate and
f_hat the MLMC PDF estimate.

Used for both the pilot MH chain (the marginal-gain race's measurement instrument) and
the production chain (the final correction term).

Includes optional numba-compiled fast paths for the grid-based (build_pi_hat) and
GNR densities, and a bisection tuning for the proposal step size.
"""

import math
from collections.abc import Callable

import numpy as np

__all__ = ["build_pi_hat", "build_log_pi_hat_from_callables", "grw_metropolis_hastings",
           "estimate_acceptance_rate", "tune_step_size"]

try:
    from numba import njit
    _NUMBA = True
    # numba optional: without it, grw_metropolis_hastings uses its Python loop
except ImportError:
    _NUMBA = False

    def njit(*args: object, **kwargs: object) -> Callable:
        """No-op decorator stand-in for numba.njit, used when numba is unavailable."""
        def _wrap(fn: Callable) -> Callable:
            """Return `fn` unchanged (no compilation without numba)."""
            return fn
        return _wrap(args[0]) if args and callable(args[0]) else _wrap

@njit(cache=True)
def _interp_njit(x: float, grid: np.ndarray, vals: np.ndarray, left: float, right: float) -> float:
    """Scalar linear interpolation on a sorted grid, matching np.interp (with left/right
    clamp values for x outside the grid range)."""

    n = grid.shape[0]
    if x <= grid[0]:
        return vals[0] if x == grid[0] else left
    if x >= grid[-1]:
        return vals[-1] if x == grid[-1] else right
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    frac = (x - grid[lo]) / (grid[hi] - grid[lo])
    return vals[lo] * (1.0 - frac) + vals[hi] * frac

@njit(cache=True)
def _log_pi_hat_njit(x: float, S_0: float, S_1: float, g_cdf: np.ndarray, F_vals: np.ndarray,
                     g_pdf: np.ndarray, f_vals: np.ndarray, a: float, b: float,
                     log_beta_norm: float) -> float:
    """Compiled scalar log target density (mirrors log_pi_hat's scalar fast-path)."""

    if x < S_0 or x > S_1:
        return -np.inf
    u = _interp_njit(x, g_cdf, F_vals, 0.0, 1.0)
    if u < 1e-300:
        u = 1e-300
    elif u > 1.0 - 1e-12:
        u = 1.0 - 1e-12
    fval = _interp_njit(x, g_pdf, f_vals, 0.0, 0.0)
    if fval <= 0.0:
        return -np.inf
    if fval < 1e-300:
        fval = 1e-300
    log_g = (a - 1.0) * np.log(u) + (b - 1.0) * np.log1p(-u) - log_beta_norm
    return log_g + np.log(fval)

@njit(cache=True)
def _grw_loop_njit(x1: float, s: float, n_samples: int, burn_in: int, thin: int,
                   noise: np.ndarray, log_u: np.ndarray,
                   S_0: float, S_1: float, g_cdf: np.ndarray, F_vals: np.ndarray,
                   g_pdf: np.ndarray, f_vals: np.ndarray, a: float, b: float,
                   log_beta_norm: float) -> tuple[np.ndarray, int]:
    """Compiled Gaussian random-walk MH loop over pre-drawn noise/log_u."""

    total_steps = burn_in + n_samples * thin
    x = x1
    log_pi_x = _log_pi_hat_njit(x, S_0, S_1, g_cdf, F_vals, g_pdf, f_vals, a, b, log_beta_norm)
    samples = np.empty(n_samples)
    n_accept = 0
    idx = 0
    for t in range(total_steps):
        x_prop = x + s * noise[t]
        log_pi_prop = _log_pi_hat_njit(x_prop, S_0, S_1, g_cdf, F_vals, g_pdf, f_vals,
                                       a, b, log_beta_norm)
        if log_u[t] < log_pi_prop - log_pi_x:
            x = x_prop
            log_pi_x = log_pi_prop
            n_accept += 1
        if t >= burn_in and (t - burn_in) % thin == 0:
            samples[idx] = x
            idx += 1
    return samples, n_accept


def _bary_weights(degree: int) -> np.ndarray:
    """Weights for equidistant nodes (grid-independent for an equidistant grid, so computed
    once and reused per segment)."""

    return np.array([(-1.0) ** j * math.comb(degree, j) for j in range(degree + 1)],
                    dtype=np.float64)

def _uniform_grid_params(grid: np.ndarray) -> tuple[bool, float, float]:
    """Return (is_uniform, s0, dstep) for a 1-D grid (GNR interpolate at equidistant points)."""

    grid = np.asarray(grid, dtype=float)
    if grid.shape[0] < 2:
        return False, 0.0, 0.0
    diffs = np.diff(grid)
    dstep = diffs.mean()
    is_uniform = bool(np.all(np.abs(diffs - dstep) <= 1e-9 * max(abs(dstep), 1.0)))
    return is_uniform, float(grid[0]), float(dstep)

@njit(cache=True)
def _bary_seg_njit(x: float, s_grid: np.ndarray, y: np.ndarray, i0: int, degree: int,
                   bary_w: np.ndarray) -> float:
    """Lagrange value of the degree-`degree` polynomial through knots
    s_grid[i0 : i0+degree+1] at x (O(degree); same polynomial as _lagrange_eval)."""

    num = 0.0
    den = 0.0
    for j in range(degree + 1):
        diff = x - s_grid[i0 + j]
        if diff == 0.0:
            return y[i0 + j]
        t = bary_w[j] / diff
        num += t * y[i0 + j]
        den += t
    return num / den

@njit(cache=True)
def _gnr_pdf_njit(x: float, p_s0: float, p_dstep: float, p_nseg: int, s_pdf: np.ndarray,
                  rho_vals: np.ndarray, degree: int, bary_w: np.ndarray) -> float:
    """Compiled GNR density: constant-extrapolation clip, O(1) equidistant segment index,
    barycentric degree-r Lagrange, non-negativity floor."""

    lo = s_pdf[0]
    hi = s_pdf[s_pdf.shape[0] - 1]
    if x < lo:
        x = lo
    elif x > hi:
        x = hi
    seg = int((x - p_s0) / (degree * p_dstep))
    if seg < 0:
        seg = 0
    elif seg > p_nseg - 1:
        seg = p_nseg - 1
    val = _bary_seg_njit(x, s_pdf, rho_vals, degree * seg, degree, bary_w)
    return val if val > 0.0 else 0.0

@njit(cache=True)
def _gnr_cdf_njit(x: float, d_s0: float, d_dstep: float, corrected: np.ndarray,
                  n_dense: int) -> float:
    """Compiled GNR CDF: Linear lookup over the dense Q_k^r sampling, constant
    extrapolation."""
    if x <= d_s0:
        return corrected[0]
    hi = d_s0 + (n_dense - 1) * d_dstep
    if x >= hi:
        return corrected[n_dense - 1]
    t = (x - d_s0) / d_dstep
    i = int(t)
    if i >= n_dense - 1:
        return corrected[n_dense - 1]
    frac = t - i
    return corrected[i] * (1.0 - frac) + corrected[i + 1] * frac

@njit(cache=True)
def _log_pi_hat_gnr_njit(x: float, S_0: float, S_1: float, d_s0: float, d_dstep: float,
                         corrected: np.ndarray, n_dense: int,
                         p_s0: float, p_dstep: float, p_nseg: int, s_pdf: np.ndarray,
                         rho_vals: np.ndarray, degree: int, bary_w: np.ndarray,
                         a: float, b: float, log_beta_norm: float) -> float:
    """Compiled scalar log target density using the GNR-interpolant CDF/PDF path."""

    if x < S_0 or x > S_1:
        return -np.inf
    u = _gnr_cdf_njit(x, d_s0, d_dstep, corrected, n_dense)
    if u < 1e-300:
        u = 1e-300
    elif u > 1.0 - 1e-12:
        u = 1.0 - 1e-12
    fval = _gnr_pdf_njit(x, p_s0, p_dstep, p_nseg, s_pdf, rho_vals, degree, bary_w)
    if fval <= 0.0:
        return -np.inf
    if fval < 1e-300:
        fval = 1e-300
    log_g = (a - 1.0) * np.log(u) + (b - 1.0) * np.log1p(-u) - log_beta_norm
    return log_g + np.log(fval)

@njit(cache=True)
def _grw_loop_gnr_njit(x1: float, s: float, n_samples: int, burn_in: int, thin: int,
                       noise: np.ndarray, log_u: np.ndarray,
                       S_0: float, S_1: float, d_s0: float, d_dstep: float,
                       corrected: np.ndarray, n_dense: int,
                       p_s0: float, p_dstep: float, p_nseg: int, s_pdf: np.ndarray,
                       rho_vals: np.ndarray, degree: int, bary_w: np.ndarray,
                       a: float, b: float, log_beta_norm: float) -> tuple[np.ndarray, int]:
    """Compiled Gaussian random-walk MH loop with the GNR density inlined."""

    total_steps = burn_in + n_samples * thin
    x = x1
    log_pi_x = _log_pi_hat_gnr_njit(x, S_0, S_1, d_s0, d_dstep, corrected, n_dense,
                                    p_s0, p_dstep, p_nseg, s_pdf, rho_vals, degree, bary_w,
                                    a, b, log_beta_norm)
    samples = np.empty(n_samples)
    n_accept = 0
    idx = 0
    for t in range(total_steps):
        x_prop = x + s * noise[t]
        log_pi_prop = _log_pi_hat_gnr_njit(x_prop, S_0, S_1, d_s0, d_dstep, corrected, n_dense,
                                           p_s0, p_dstep, p_nseg, s_pdf, rho_vals, degree,
                                           bary_w, a, b, log_beta_norm)
        if log_u[t] < log_pi_prop - log_pi_x:
            x = x_prop
            log_pi_x = log_pi_prop
            n_accept += 1
        if t >= burn_in and (t - burn_in) % thin == 0:
            samples[idx] = x
            idx += 1
    return samples, n_accept

# build pi_hat, F_hat, f_hat from MLMC grid output

def build_log_pi_hat_from_callables(a: float, b: float,
                                     F_hat_fn: Callable[[np.ndarray], np.ndarray],
                                     f_hat_fn: Callable[[np.ndarray], np.ndarray],
                                     S_0: float, S_1: float) -> Callable[[np.ndarray], np.ndarray]:
    """Construct log_pi_hat(x) = log g_{a,b}(F_hat_fn(x)) + log f_hat_fn(x) from already built
    F_hat_fn / f_hat_fn callables.

    Points outside [S_0, S_1] give log_pi_hat = -inf.
    """

    log_beta_norm = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def log_pi_hat(x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the log target density, scalar or vectorized."""

        if np.ndim(x) == 0:
            xf = float(x)
            if xf < S_0 or xf > S_1:
                return -np.inf
            u = min(max(float(F_hat_fn(xf)), 1e-300), 1.0 - 1e-12)
            fval = float(f_hat_fn(xf))
            if fval <= 0.0:
                return -np.inf
            log_g = (a - 1.0) * math.log(u) + (b - 1.0) * math.log1p(-u) - log_beta_norm
            return log_g + math.log(max(fval, 1e-300))

        x = np.asarray(x, dtype=float)
        out = np.full_like(x, -np.inf, dtype=float)
        inside = (x >= S_0) & (x <= S_1)

        if np.any(inside):
            u = F_hat_fn(x[inside])
            u = np.clip(u, 1e-300, 1.0 - 1e-12)
            log_g = (a - 1.0) * np.log(u) + (b - 1.0) * np.log1p(-u) - log_beta_norm

            f = f_hat_fn(x[inside])
            log_f = np.where(f > 0.0, np.log(np.clip(f, 1e-300, None)), -np.inf)

            out[inside] = log_g + log_f

        return out

    # Fully-compiled fast path
    cdf_data = getattr(F_hat_fn, "_gnr_cdf_data", None)
    pdf_data = getattr(f_hat_fn, "_gnr_pdf_data", None)
    if _NUMBA and cdf_data is not None and pdf_data is not None:
        dense_x, corrected = cdf_data
        s_pdf, rho_vals, degree = pdf_data
        uni_c, d_s0, d_dstep = _uniform_grid_params(dense_x)
        uni_p, p_s0, p_dstep = _uniform_grid_params(s_pdf)
        if uni_c and uni_p and degree >= 1 and (len(s_pdf) - 1) % degree == 0:
            log_pi_hat._njit_gnr_data = (
                float(S_0), float(S_1),
                d_s0, d_dstep, np.ascontiguousarray(corrected, dtype=np.float64), len(dense_x),
                p_s0, p_dstep, (len(s_pdf) - 1) // degree,
                np.ascontiguousarray(s_pdf, dtype=np.float64),
                np.ascontiguousarray(rho_vals, dtype=np.float64),
                int(degree), _bary_weights(int(degree)),
                float(a), float(b), float(log_beta_norm),
            )

    return log_pi_hat


def build_pi_hat(a: float, b: float, s_grid_cdf: np.ndarray, F_vals: np.ndarray,
                 s_grid_pdf: np.ndarray, f_vals: np.ndarray, S_0: float, S_1: float
                 ) -> tuple[Callable[[np.ndarray], np.ndarray],
                            Callable[[np.ndarray], np.ndarray],
                            Callable[[np.ndarray], np.ndarray]]:
    """Construct the callables F_hat, f_hat, log_pi_hat used by the MH sampler and by the
    marginal-gain calculations.

    F_hat and f_hat are built by linear interpolation on the MLMC output grids. Points
    outside [S_0, S_1] give F_hat = 0 or 1 (clamped) and f_hat = 0, so log_pi_hat = -inf
    there.
    """

    def F_hat(x: np.ndarray) -> np.ndarray:
        """Vectorized CDF evaluator via linear interpolation on s_grid_cdf/F_vals."""

        return np.interp(x, s_grid_cdf, F_vals, left=0.0, right=1.0)

    def f_hat(x: np.ndarray) -> np.ndarray:
        """Vectorized PDF evaluator via linear interpolation on s_grid_pdf/f_vals."""

        return np.interp(x, s_grid_pdf, f_vals, left=0.0, right=0.0)

    log_pi_hat = build_log_pi_hat_from_callables(a, b, F_hat, f_hat, S_0, S_1)


    if _NUMBA:
        log_beta_norm = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        log_pi_hat._njit_data = (
            float(S_0), float(S_1),
            np.ascontiguousarray(s_grid_cdf, dtype=np.float64),
            np.ascontiguousarray(F_vals, dtype=np.float64),
            np.ascontiguousarray(s_grid_pdf, dtype=np.float64),
            np.ascontiguousarray(f_vals, dtype=np.float64),
            float(a), float(b), float(log_beta_norm),
        )

    return F_hat, f_hat, log_pi_hat



def grw_metropolis_hastings(log_pi_hat: Callable[[float], float], x1: float, s: float,
                             n_samples: int, burn_in: int | None = None, thin: int = 1,
                             rng: np.random.Generator | None = None
                             ) -> tuple[np.ndarray, float]:
    """Gaussian random-walk MH sampler targeting exp(log_pi_hat)."""

    if rng is None:
        rng = np.random.default_rng()
    if burn_in is None:
        burn_in = n_samples // 5

    log_pi_x1 = log_pi_hat(x1)
    if not np.isfinite(log_pi_x1):
        raise ValueError(
            f"MH starting point x1={x1} has log_pi_hat = {log_pi_x1} "
            f"(outside support or zero density). Choose a different x1")

    total_steps = burn_in + n_samples * thin

    # Pre-draw all randomness to improve the speed
    proposals_noise = rng.standard_normal(total_steps)
    log_u = np.log(rng.uniform(size=total_steps))

    # Compiled fast path
    gnr_data = getattr(log_pi_hat, "_njit_gnr_data", None)
    if gnr_data is not None:
        samples, n_accept = _grw_loop_gnr_njit(float(x1), float(s), n_samples, burn_in, thin,
                                               proposals_noise, log_u, *gnr_data)
        return samples, n_accept / total_steps

    # Compiled fast path when the density has raw grid data 
    njit_data = getattr(log_pi_hat, "_njit_data", None)
    if njit_data is not None:
        samples, n_accept = _grw_loop_njit(float(x1), float(s), n_samples, burn_in, thin,
                                           proposals_noise, log_u, *njit_data)
        return samples, n_accept / total_steps

    x = x1
    log_pi_x = log_pi_x1
    samples = np.empty(n_samples, dtype=float)
    n_accept = 0
    idx = 0

    for t in range(total_steps):
        x_prop = x + s * proposals_noise[t]
        log_pi_prop = log_pi_hat(x_prop)
        log_alpha = log_pi_prop - log_pi_x

        if log_u[t] < log_alpha:
            x = x_prop
            log_pi_x = log_pi_prop
            n_accept += 1

        if t >= burn_in and (t - burn_in) % thin == 0:
            samples[idx] = x
            idx += 1

    acc_rate = n_accept / total_steps
    return samples, acc_rate

# MH proposal step-size tuning: bisection on the observed acceptance rate

TARGET_AR = 0.44 # (See Gelman Roberts Gilks 1996)
AR_TOL = 0.03
PILOT_N = 2000
MAX_ITER = 20


def estimate_acceptance_rate(step_size: float, log_pi_hat: Callable[[float], float], x1: float,
                             n: int = PILOT_N, rng: np.random.Generator | None = None) -> float:
    """Run a short pilot MH chain at `step_size` and return its observed acceptance rate."""

    if rng is None:
        rng = np.random.default_rng()
    _, ar = grw_metropolis_hastings(log_pi_hat, x1, step_size, n,
                                     burn_in=n // 5, rng=rng)
    return ar


def tune_step_size(log_pi_hat: Callable[[float], float], x1: float,
                    step_lo: float | None = None, step_hi: float | None = None,
                    target_ar: float = TARGET_AR, ar_tol: float = AR_TOL,
                    pilot_n: int = PILOT_N, max_iter: int = MAX_ITER,
                    initial_guess: float | None = None,
                    rng: np.random.Generator | None = None,
                    verbose: bool = False) -> tuple[float, float]:
    """Bisection search for the MH proposal step size hitting target_ar."""

    if rng is None:
        rng = np.random.default_rng()

    if initial_guess is not None and initial_guess > 0:
        step_lo = initial_guess * 0.1 if step_lo is None else step_lo
        step_hi = initial_guess * 10.0 if step_hi is None else step_hi
    else:
        step_lo = 1e-3 if step_lo is None else step_lo
        step_hi = 5.0 if step_hi is None else step_hi

    ar_lo = estimate_acceptance_rate(step_lo, log_pi_hat, x1, pilot_n, rng)
    ar_hi = estimate_acceptance_rate(step_hi, log_pi_hat, x1, pilot_n, rng)
    if verbose:
        print(f"  Initial bracket: step={step_lo:.5g} -> AR={ar_lo:.2%}, "
              f"step={step_hi:.5g} -> AR={ar_hi:.2%}")

    if not (ar_hi <= target_ar <= ar_lo):
        if verbose:
            print(f"  WARNING: target AR={target_ar:.0%} not restricted. Widening...")
        step_lo, step_hi = step_lo * 1e-2, step_hi * 1e2
        ar_lo = estimate_acceptance_rate(step_lo, log_pi_hat, x1, pilot_n, rng)
        ar_hi = estimate_acceptance_rate(step_hi, log_pi_hat, x1, pilot_n, rng)
        if verbose:
            print(f"  New bracket: step={step_lo:.5g} -> AR={ar_lo:.2%}, "
                  f"step={step_hi:.5g} -> AR={ar_hi:.2%}")
        if not (ar_hi <= target_ar <= ar_lo):
            return (step_lo, ar_lo) if abs(ar_lo - target_ar) < abs(ar_hi - target_ar) else (step_hi, ar_hi)

    step_mid, ar_mid = step_lo, ar_lo
    for i in range(max_iter):
        step_mid = (step_lo + step_hi) / 2
        ar_mid = estimate_acceptance_rate(step_mid, log_pi_hat, x1, pilot_n, rng)
        if verbose:
            print(f"  [{i+1:2d}] step={step_mid:.5g}  AR={ar_mid:.2%}")
        if abs(ar_mid - target_ar) < ar_tol:
            return step_mid, ar_mid
        if ar_mid > target_ar:
            step_lo = step_mid
        else:
            step_hi = step_mid

    return step_mid, ar_mid

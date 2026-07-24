"""Module providing fine-only baseline quantile estimators (Harrell-Davis and order-statistic).

Each estimates a quantile directly from a sorted batch of fine-model samples. They are
the reference points the BL-HD estimator is compared against at equal cost. Both return
the raw signed quantile at level alpha.
"""

import numpy as np
from scipy.special import betainc

__all__ = ["harrell_davis", "order_statistic_quantile"]

def harrell_davis(sorted_samples: np.ndarray, alpha: float) -> float:
    """Harrell-Davis quantile estimate at level alpha.

    q_HD = sum_i W_i * x_(i) with W_i = I_{i/n}(a, b) - I_{(i-1)/n}(a, b),
    a = alpha*(n+1), b = (1-alpha)*(n+1), I the regularized incomplete beta function.
    """
    n = len(sorted_samples)
    a = alpha * (n + 1)
    b = (1 - alpha) * (n + 1)
    grid = np.arange(0, n + 1) / n
    weights = np.diff(betainc(a, b, grid))
    return float(np.dot(weights, sorted_samples))

def order_statistic_quantile(sorted_samples: np.ndarray, alpha: float) -> float:
    """Linear-interpolation order-statistic quantile at level alpha."""
    n = len(sorted_samples)
    h = (n - 1) * alpha
    lo = int(np.floor(h))
    hi = min(lo + 1, n - 1)
    frac = h - lo
    return sorted_samples[lo] + frac * (sorted_samples[hi] - sorted_samples[lo])

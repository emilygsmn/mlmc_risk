"""Module providing nalytical coupled fine/coarse samplers for bivariate Gaussian and
bivariate Student-t.

Illustrative test models where the fine (L_f) and coarse (L_dg) losses share a marginal
distribution and a controllable dependence, used to study the estimator beyond the
portfolio.

The first component is interpreted as the true loss and the second as an approximation.
This model allows for direct control over the correlation and has analytically
tractable quantiles.
"""

from collections.abc import Callable

import numpy as np

from mlmc_risk_estimation.blhd.samplers.base import make_sampler

__all__ = ["normal_sampler", "studentt_sampler"]

def normal_sampler(rho: float, mu: float = 0.0, sd: float = 1.0) -> dict[str, Callable]:
    """Bivariate-normal fine/coarse model (L_f, L_c) with correlation rho.
    """

    def level1(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw n coupled (fine, coarse) pairs and their difference."""
        z1 = np.random.normal(0.0, 1.0, n)
        z2 = np.random.normal(0.0, 1.0, n)
        fine = mu + sd * z1
        coarse = mu + sd * (rho * z1 + np.sqrt(1.0 - rho ** 2) * z2)
        return fine, coarse, coarse - fine

    def level0(n: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw n coarse-only samples for MLMC level 0."""
        coarse = np.random.normal(mu, sd, n)
        return coarse, np.zeros(n)

    def coarse(n: int) -> np.ndarray:
        """Draw n iid coarse (DG) samples."""
        return np.random.normal(mu, sd, n)

    def fine(n: int) -> np.ndarray:
        """Draw n iid fine samples."""
        return np.random.normal(mu, sd, n)

    return make_sampler(fine=fine, coarse=coarse, level0=level0, level1=level1)

def studentt_sampler(rho: float, dof: float) -> dict[str, Callable]:
    """Bivariate Student-t fine/coarse model: dof degrees of freedom, Gaussian-copula
    correlation rho, location 0, scale 1 marginals.
    """

    def level1(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw n coupled (fine, coarse) pairs sharing a single chi-squared mixing variable."""
        z1 = np.random.normal(0.0, 1.0, n)
        z2 = np.random.normal(0.0, 1.0, n)
        denom = np.sqrt(np.random.chisquare(dof, n) / dof)
        fine = z1 / denom
        coarse = (rho * z1 + np.sqrt(1.0 - rho ** 2) * z2) / denom
        return fine, coarse, coarse - fine

    def level0(n: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw n coarse-only samples for MLMC level 0."""
        z = np.random.normal(0.0, 1.0, n)
        coarse = z / np.sqrt(np.random.chisquare(dof, n) / dof)
        return coarse, np.zeros(n)

    def coarse(n: int) -> np.ndarray:
        """Draw n iid coarse (DG) samples."""
        return level0(n)[0]

    def fine(n: int) -> np.ndarray:
        """Draw n iid fine samples."""
        z = np.random.normal(0.0, 1.0, n)
        return z / np.sqrt(np.random.chisquare(dof, n) / dof)

    return make_sampler(fine=fine, coarse=coarse, level0=level0, level1=level1)

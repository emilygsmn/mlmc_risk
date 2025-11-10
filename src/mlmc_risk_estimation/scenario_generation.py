""""Module providing functions for Monte Carlo real-world scenario generation."""

import numpy as np
from scipy import stats

__all__ = ["generate_mc_scenarios"]

def _calc_correlation_mat(prices):
    """Function calculating the correlation matrix from a time series of prices."""

    # Compute the daily returns
    daily_returns = prices.pct_change().dropna()

    return daily_returns.corr()

def _calc_cholesky_mat(mat):
    """Function calculating the cholesky decomposition for a given matrix."""

    return np.linalg.cholesky(mat)

def _correlate_scenarios(uncorr_samples, corr_mat):
    """Function that introduces correlation to the MC scenarios."""

    # Calculate the Cholesky decomposition of the correlation matrix
    cholesky_mat = _calc_cholesky_mat(corr_mat)

    return uncorr_samples @ cholesky_mat.T

def generate_mc_scenarios(market_data, num_scen):
    """Function generating real-world Monte Carlo scenarios for all the risk factors."""

    # Generate num_scen independent (uncorrelated) samples from Unif[0,1] for all RFs
    num_rfs = len(market_data.columns)
    uncorr_unif_samples = np.random.rand(num_scen, num_rfs)

    # Use inverse transformation to get independent (uncorrelated) samples from N(0,1)
    uncorr_normal_samples = stats.norm.ppf(uncorr_unif_samples)

    # Correlate the samples (assumption: Gaussian copula)
    corr_mat = _calc_correlation_mat(market_data)
    corr_normal_samples = _correlate_scenarios(uncorr_normal_samples, corr_mat)

    return corr_normal_samples

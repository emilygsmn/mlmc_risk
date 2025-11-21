""""Module providing functions for Monte Carlo real-world scenario generation."""

import numpy as np
from scipy import stats
import pandas as pd

__all__ = ["generate_mc_shocks", "generate_mc_shocks_pycopula"]

def _calc_correlation_mat(prices):
    """Function calculating the correlation matrix from a time series of prices."""

    # Compute the daily returns
    daily_returns = prices.pct_change().dropna()

    return daily_returns.corr()

def _calc_cholesky_mat(mat):
    """Function calculating the cholesky decomposition for a given matrix."""

    return np.linalg.cholesky(mat)

def _correlate_scenarios(uncorr_samples, corr_mat, rfs):
    """Function that introduces correlation to the MC scenarios."""

    # Calculate the Cholesky decomposition of the correlation matrix
    cholesky_mat = _calc_cholesky_mat(corr_mat)

    # Correlate the samples
    corr_samples = uncorr_samples @ cholesky_mat.T

    return pd.DataFrame(corr_samples, columns=rfs)

def generate_mc_shocks(market_data, marg_distr_map, calib_param, num_scen):
    """Function generating real-world Monte Carlo scenarios for all the risk factors."""

    # Generate num_scen independent (uncorrelated) samples from Unif[0,1] for all RFs
    rfs = market_data.columns
    num_rfs = len(rfs)
    uncorr_unif_samples = np.random.rand(num_scen, num_rfs)

    # Use inverse transformation to get independent (uncorrelated) samples from N(0,1)
    uncorr_normal_samples = stats.norm.ppf(uncorr_unif_samples)

    # Correlate the samples (assumption: Gaussian copula)
    corr_mat = _calc_correlation_mat(market_data)
    corr_normal_samples = _correlate_scenarios(uncorr_normal_samples, corr_mat, rfs)

    return _map_to_marginals(corr_normal_samples, marg_distr_map, calib_param)

########## Scenario Generation using NumPy.Random.multivariate_normal() ##########

def _sample_from_copula(corr_mat, rfs, num_scen):
    """Function to generate samples from a given copula."""

    # Generate num_scen samples from the copula
    norm_samples = np.random.multivariate_normal(mean=np.zeros(len(rfs)), cov=corr_mat, size=num_scen)

    return pd.DataFrame(norm_samples, columns=rfs)

def _calc_shock_with_bm(x, mu, sigma, dt):
    """Function calculating the MC scenario shocks for the RFs using Brownian Motion"""
    return mu * dt + sigma * np.sqrt(dt) * x

def _calc_shock_with_gbm(x, mu, sigma, dt):
    """Function calculating the MC scenario shocks for the RFs using Geometrical Brownian Motion"""
    return np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * x) - 1

def _calc_shock_with_sgbm(x, mu, sigma, dt, shift):
    """Function calculating the MC scenario shocks for the RFs using Shifted Geom. Brown. Motion"""

    gbm = _calc_shock_with_gbm(x, mu, sigma, dt)

    return (1 + shift) * gbm - shift

def _map_to_marginals(samples, marg_distr_map, param):
    """Function to map the correlated uniform MC samples to their marginal shock distributions."""

    shock_functions = {
        "BM": _calc_shock_with_bm,
        "Geom_BM": _calc_shock_with_gbm,
        "Shift_Geom_BM": _calc_shock_with_sgbm
    }

    mc_shocks = pd.DataFrame(columns=samples.columns, index=samples.index)

    for col in samples.columns:
        key = col[:2]
        model_type = marg_distr_map[key]
        shock_fun = shock_functions[model_type]
        param = param[key]

        # Pick correct call signature
        if model_type == "Shift_Geom_BM":
            mc_shocks[col] = shock_fun(samples[col],
                                        mu=param["mu"],
                                        sigma=param["sigma"],
                                        dt=param["dt"],
                                        shift=param["shift"])
        else:
            mc_shocks[col] = shock_fun(samples[col],
                                        mu=param["mu"],
                                        sigma=param["sigma"],
                                        dt=param["dt"])

    return mc_shocks

def generate_mc_shocks_pycopula(param_config, market_data, calib_param):
    """Function generating real-world Monte Carlo scenarios for all risk factors."""

    # Get the number of risk factors and their names
    rfs = list(market_data.columns)

    # Calculate the correlation matrix
    corr_mat = _calc_correlation_mat(market_data)

    # Sample from the copula num_scen times
    corr_normal_samples = _sample_from_copula(corr_mat=corr_mat,
                                              rfs=rfs,
                                              num_scen=param_config["monte_carlo"]["n"]
                                              )

    return _map_to_marginals(samples=corr_normal_samples,
                             marg_distr_map=param_config["valuation"]["stoch_proc_map"],
                             param=calib_param
                             )

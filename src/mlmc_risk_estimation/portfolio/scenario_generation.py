"""Module providing functions for Monte Carlo real-world scenario generation."""

import warnings

import numpy as np
import pandas as pd

__all__ = ["generate_mc_shocks_pycopula"]

def _calc_factor_returns(prices: pd.DataFrame,
                         instr_info: pd.DataFrame,
                         return_type_map: dict[str, str],
                         shift: float
                         ) -> pd.DataFrame:
    """Compute each risk factor's daily returns using its configured return type."""

    returns = pd.DataFrame(index=prices.index)
    for col in prices.columns:
        if col.startswith("IR"):
            # Shifted-log returns keep the log defined for rates that are negative or near zero
            returns[col] = np.log(prices[col] + shift).diff()
            continue

        if col.startswith("EQVOL"):
            val_tag = "EQVOL"
        else:
            val_tag = instr_info.loc[instr_info["fin_instr"] == col, "val_tag"].iloc[0]

        if return_type_map[val_tag] == "rel":
            returns[col] = prices[col].pct_change(fill_method=None)
        else:
            returns[col] = prices[col].diff()

    return returns.dropna()

def _calc_correlation_mat(prices: pd.DataFrame,
                          instr_info: pd.DataFrame,
                          return_type_map: dict[str, str],
                          shift: float
                          ) -> np.ndarray:
    """Calculate the correlation matrix from per-factor daily returns."""

    returns = _calc_factor_returns(prices, instr_info, return_type_map, shift)
    corr_mat = returns.corr().values

    if not _check_corr_matrix_is_spd(corr_mat):
        warnings.warn("Correlation matrix is not symmetric positive-definite. "
                      "Applying jitter to restore positive-definiteness.")
        return _add_jitter(corr_mat)

    return corr_mat

def _check_corr_matrix_is_spd(mat: np.ndarray) -> bool:
    """Check whether a matrix is symmetric positive-definite via Cholesky."""
    try:
        np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False

def _add_jitter(mat: np.ndarray, eps: float = 1e-8, max_tries: int = 5) -> np.ndarray:
    """Repair a singular correlation matrix by eigenvalue clipping:
    Diagonalizes (symmetric eigendecomposition), fix all eigenvalues below eps at eps, then 
    reconstruct by undoing the diagonalization. The reconstruction generally perturbs the
    diagonal away from 1, so it is rescaled back to an exact unit diagonal afterward. This is
    required for the result to still be a valid correlation matrix (not just any SPD matrix).
    """
    mat = np.array(mat, dtype=float).copy()
    mat = (mat + mat.T) / 2

    for _ in range(max_tries):
        eigvals, eigvecs = np.linalg.eigh(mat)
        clipped = np.maximum(eigvals, eps)
        repaired = eigvecs @ np.diag(clipped) @ eigvecs.T
        repaired = (repaired + repaired.T) / 2

        # Rescale to restore an exact unit diagonal (still a valid correlation matrix)
        d = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(d, d)
        np.fill_diagonal(repaired, 1.0)

        # If the floor was not enough to survive rescaling, tighten and retry
        if _check_corr_matrix_is_spd(repaired):
            return repaired
        eps *= 10

    raise np.linalg.LinAlgError(
        f"Failed to repair correlation matrix to SPD via eigenvalue clipping after {max_tries} "
        "attempts.")

def _calc_cholesky_mat(mat: np.ndarray) -> np.ndarray:
    """Calculate the cholesky decomposition for a given matrix."""

    return np.linalg.cholesky(mat)

def _correlate_scenarios(uncorr_samples: np.ndarray,
                         corr_mat: np.ndarray,
                         rfs: list[str]
                         ) -> pd.DataFrame:
    """Introduce correlation to the MC scenarios."""

    cholesky_mat = _calc_cholesky_mat(corr_mat)
    corr_samples = uncorr_samples @ cholesky_mat.T

    return pd.DataFrame(corr_samples, columns=rfs)

def _sample_from_copula(corr_mat: np.ndarray, rfs: list[str], num_scen: int) -> pd.DataFrame:
    """Generate samples from a given copula."""

    norm_samples = np.random.multivariate_normal(mean=np.zeros(len(rfs)),
                                                 cov=corr_mat,
                                                 size=num_scen
                                                 )

    return pd.DataFrame(norm_samples, columns=rfs)

def _calc_shock_with_bm(x: np.ndarray, mu: float, sigma: float, dt: float) -> np.ndarray:
    """Calculate the MC scenario shocks for the RFs using Brownian Motion."""

    return mu * dt + sigma * np.sqrt(dt) * x

def _calc_shock_with_gbm(x: np.ndarray, mu: float, sigma: float, dt: float) -> np.ndarray:
    """Calculate the MC scenario shocks for the RFs using Geometrical Brownian Motion."""

    return np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * x) - 1

def _calc_shock_with_sgbm(x: np.ndarray,
                          mu: float,
                          sigma: float,
                          dt: float,
                          shift: float,
                          base_level: float
                          ) -> np.ndarray:
    """Calculate the MC scenario shocks for the RFs using Shifted Geom. Brown. Motion."""

    gbm_shock = _calc_shock_with_gbm(x, mu, sigma, dt)

    return (base_level + shift) * gbm_shock

def _map_to_marginals(samples: pd.DataFrame,
                      marg_distr_map: dict[str, str],
                      instr_info: pd.DataFrame,
                      param: pd.DataFrame,
                      base_levels: pd.Series,
                      shift: float = 0.035
                      ) -> pd.DataFrame:
    """Map the correlated uniform MC samples to their marginal shock distributions."""

    shock_functions = {
        "BM": _calc_shock_with_bm,
        "Geom_BM": _calc_shock_with_gbm,
        "Shift_Geom_BM": _calc_shock_with_sgbm
    }

    mc_shocks = pd.DataFrame(columns=samples.columns, index=samples.index)

    for col in samples.columns:
        if col.startswith("IR"):
            val_tag = "IR"
        elif col.startswith("EQVOL"):
            val_tag = "EQVOL"
        else:
            val_tag = instr_info.loc[instr_info["fin_instr"] == col, "val_tag"].iloc[0]
        model_type = marg_distr_map[val_tag]
        shock_fun = shock_functions[model_type]

        sigma = param.loc["sigma", col]

        # No drift: for log-normal models this means mu = 0.5*sigma^2, which cancels the
        # -0.5*sigma^2 Ito term in the shock exponent. Standard BM has no such term, so mu=0.
        if model_type in ("Geom_BM", "Shift_Geom_BM"):
            mu = 0.5 * sigma ** 2
        else:
            mu = 0

        if model_type == "Shift_Geom_BM":
            mc_shocks[col] = shock_fun(samples[col],
                                        mu=mu,
                                        sigma=sigma,
                                        dt=1,
                                        shift=shift,
                                        base_level=base_levels[col])
        else:
            mc_shocks[col] = shock_fun(samples[col],
                                        mu=mu,
                                        sigma=sigma,
                                        dt=1)

    return mc_shocks

def generate_mc_shocks_pycopula(market_data: pd.DataFrame,
                                instr_info: pd.DataFrame,
                                param_config: dict,
                                calib_param: pd.DataFrame,
                                ref_date: str
                                ) -> pd.DataFrame:
    """Generate real-world Monte Carlo scenarios for all risk factors."""

    rfs = list(market_data.columns)
    corr_mat = _calc_correlation_mat(
        market_data,
        instr_info,
        param_config["valuation"]["calibr_methods"]["return_type"],
        param_config["valuation"]["sgbm_shift"]
        )

    corr_normal_samples = _sample_from_copula(corr_mat=corr_mat,
                                              rfs=rfs,
                                              num_scen=param_config["monte_carlo"]["n"]
                                              )

    return _map_to_marginals(samples=corr_normal_samples,
                             marg_distr_map=param_config["valuation"]["stoch_proc_map"],
                             instr_info=instr_info,
                             param=calib_param,
                             base_levels=market_data.loc[ref_date],
                             shift=param_config["valuation"]["sgbm_shift"]
                             )

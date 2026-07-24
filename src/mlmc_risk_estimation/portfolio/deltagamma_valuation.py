"""Module providing functions for delta-gamma approximate valuation
   of the risk factors."""

import numpy as np
import pandas as pd

from mlmc_risk_estimation.portfolio.full_valuation import calc_prices
from mlmc_risk_estimation.portfolio.risk_aggregation import calc_portfolio_pnl

__all__ = ["compute_greeks", "apply_delta_pnl", "apply_delta_gamma_pnl",
           "calc_delta_scenario_pnl", "calc_delta_gamma_scenario_pnl"]

def _set_sensi_shocks(rfs: list[str], shock_types: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Create up and down shocks for each risk factor."""

    # Additive risk factors get a 1bps shock and multiplicative risk factors get a 1% shock
    ADD_SHOCK = 0.0001
    MULT_SHOCK = 0.01

    h = np.full(len(rfs), MULT_SHOCK)
    for risk_type, appl_method in shock_types.items():
        if appl_method == "add":
            mask = [risk_type in rf for rf in rfs]
            h[mask] = ADD_SHOCK

    return h.copy(), h.copy()

def _calc_delta_sensis(mkt_data: pd.DataFrame,
                       instr_info: pd.DataFrame,
                       ref_date: str,
                       param_config: dict,
                       der_underlyings: dict[str, str],
                       weights: pd.Series | None = None
                       ) -> pd.Series:
    """Calculate delta sensitivities of a portfolio."""

    rfs = list(mkt_data.columns)
    shock_types = param_config["valuation"]["shock_type"]
    h_up, h_down = _set_sensi_shocks(rfs, shock_types)
    h = h_up + h_down

    up_shocks = pd.DataFrame(np.diag(h_up), index=rfs, columns=rfs)
    down_shocks = pd.DataFrame(np.diag(-h_down), index=rfs, columns=rfs)

    up_price = calc_prices(mkt_data, instr_info, ref_date, param_config,
                           der_underlyings, up_shocks)
    down_price = calc_prices(mkt_data, instr_info, ref_date, param_config,
                             der_underlyings, down_shocks)
    price_change = calc_portfolio_pnl(up_price - down_price, weights).to_numpy().flatten()

    deltas = price_change / h

    return pd.Series(deltas, index=rfs)

def _build_diag_shock_df(rfs: list[str], h: np.ndarray) -> pd.DataFrame:
    """Build a diagonal shock DataFrame from a shock-size vector."""
    
    return pd.DataFrame(
        np.diag(h),
        index=rfs,
        columns=rfs
    )

def _calc_gamma_sensis(mkt_data: pd.DataFrame,
                       instr_info: pd.DataFrame,
                       ref_date: str,
                       param_config: dict,
                       der_underlyings: dict[str, str],
                       weights: pd.Series | None = None
                       ) -> pd.DataFrame:
    """Calculate the gamma and cross-gamma sensitivities of a portfolio.
    """

    rfs = list(mkt_data.columns)
    shock_types = param_config["valuation"]["shock_type"]
    h_up, h_down = _set_sensi_shocks(rfs, shock_types)
    h = h_up + h_down

    gamma = pd.DataFrame(0.0, index=rfs, columns=rfs)
    h_mat = pd.DataFrame(np.diag(h), index=rfs, columns=rfs)

    for i, rf_i in enumerate(rfs):
        row_i = h_mat.loc[rf_i]

        # Four shock combinations for the finite-difference cross-gamma formula
        shocks_up_up =  h_mat + row_i
        shocks_up_down =  h_mat - row_i
        shocks_down_up = -h_mat + row_i
        shocks_down_down = -h_mat - row_i

        up_up_price = (calc_portfolio_pnl(calc_prices(mkt_data,
                                                      instr_info,
                                                      ref_date,
                                                      param_config,
                                                      der_underlyings,
                                                      shocks_up_up), weights).to_numpy().ravel())
        up_down_price = (calc_portfolio_pnl(calc_prices(mkt_data,
                                                        instr_info,
                                                        ref_date,
                                                        param_config,
                                                        der_underlyings,
                                                        shocks_up_down), weights)
                                           .to_numpy().ravel())
        down_up_price = (calc_portfolio_pnl(calc_prices(mkt_data,
                                                        instr_info,
                                                        ref_date,
                                                        param_config,
                                                        der_underlyings,
                                                        shocks_down_up), weights)
                                           .to_numpy().ravel())
        down_down_price = (calc_portfolio_pnl(calc_prices(mkt_data,
                                                          instr_info,
                                                          ref_date,
                                                          param_config,
                                                          der_underlyings,
                                                          shocks_down_down), weights)
                                             .to_numpy().ravel())

        # Finite-difference formula for (cross-)gamma
        g_ij = (up_up_price - up_down_price - down_up_price + down_down_price) / (4.0 * h[i] * h)

        gamma.iloc[i, :] = g_ij
        gamma.iloc[:, i] = g_ij

    return gamma

def compute_greeks(mkt_data: pd.DataFrame,
                   instr_info: pd.DataFrame,
                   ref_date: str,
                   param_config: dict,
                   der_underlyings: dict[str, str],
                   weights: pd.Series | None = None
                   ) -> tuple[pd.Series, pd.DataFrame]:
    """Compute all first and second order portfolio sensitivities.

    The Greeks do not depend on any Monte Carlo scenario, so they can be computed once and
    reused across many scenario batches via apply_delta_pnl / apply_delta_gamma_pnl.
    """

    deltas = _calc_delta_sensis(mkt_data, instr_info, ref_date, param_config, der_underlyings,
                                weights)
    gammas = _calc_gamma_sensis(mkt_data, instr_info, ref_date, param_config, der_underlyings,
                                weights)

    return deltas, gammas

def apply_delta_pnl(deltas: pd.Series,
                    scenario_shocks: pd.DataFrame
                    ) -> pd.DataFrame:
    """Apply pre-computed deltas to scenario shocks (Delta P&L)."""

    if not deltas.index.equals(scenario_shocks.columns):
        raise ValueError("Scenario shock columns do not match delta index")

    delta = deltas.to_numpy()
    shocks = scenario_shocks.to_numpy()
    pnl_delta = shocks @ delta

    return pd.DataFrame(pnl_delta, index=scenario_shocks.index, columns=["pnl"])

def apply_delta_gamma_pnl(deltas: pd.Series,
                          gammas: pd.DataFrame,
                          scenario_shocks: pd.DataFrame
                          ) -> pd.DataFrame:
    """Apply pre-computed Greeks to scenario shocks (Delta-Gamma P&L)."""

    factors = deltas.index
    if not factors.equals(gammas.index):
        raise ValueError("Delta index and gamma index do not match")
    if not factors.equals(gammas.columns):
        raise ValueError("Gamma columns do not match delta index")
    if not factors.equals(scenario_shocks.columns):
        raise ValueError("Scenario shock columns do not match delta index")

    delta = deltas.to_numpy()
    shocks = scenario_shocks.to_numpy()
    gamma = gammas.to_numpy()

    if not np.allclose(gamma, gamma.T):
        raise ValueError("Gamma matrix must be symmetric")

    pnl_delta = shocks @ delta
    pnl_gamma = 0.5 * np.sum(shocks * (shocks @ gamma), axis=1)
    pnl_total = pnl_delta + pnl_gamma

    return pd.DataFrame(pnl_total, index=scenario_shocks.index, columns=["pnl"])

def calc_delta_scenario_pnl(mkt_data: pd.DataFrame,
                            instr_info: pd.DataFrame,
                            ref_date: str,
                            param_config: dict,
                            der_underlyings: dict[str, str],
                            scenario_shocks: pd.DataFrame,
                            weights: pd.Series | None = None
                            ) -> pd.DataFrame:
    """Calculate the scenario P&Ls approximated by Delta method."""

    deltas, _ = compute_greeks(mkt_data, instr_info, ref_date, param_config, der_underlyings,
                               weights)

    return apply_delta_pnl(deltas, scenario_shocks)

def calc_delta_gamma_scenario_pnl(mkt_data: pd.DataFrame,
                                  instr_info: pd.DataFrame,
                                  ref_date: str,
                                  param_config: dict,
                                  der_underlyings: dict[str, str],
                                  scenario_shocks: pd.DataFrame,
                                  weights: pd.Series | None = None
                                  ) -> pd.DataFrame:
    """Calculate the scenario P&Ls approximated by Delta-Gamma method."""

    deltas, gammas = compute_greeks(mkt_data, instr_info, ref_date, param_config, der_underlyings,
                                    weights)

    return apply_delta_gamma_pnl(deltas, gammas, scenario_shocks)

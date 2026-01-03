"""Module providing functions for Delta-Gamma approximate valuation
   of the risk factors."""

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from full_valuation import calc_prices
from risk_aggregation import calc_portfolio_pnl

def _set_sensi_shocks(rfs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Function creating up and down shocks for each risk factor."""

    h_up = np.full(len(rfs), 0.1)
    h_down = np.full(len(rfs), 0.01) #h_up.copy()

    return h_up, h_down

def _calc_delta_sensis(mkt_data: pd.DataFrame,
                       instr_info: pd.DataFrame,
                       ref_date: str,
                       param_config: dict,
                       der_underlyings: Dict[str, str]
                       ) -> pd.Series:
    """Function calculating delta sensitivities of a portfolio."""

    # Retrieve risk factors
    rfs = list(mkt_data.columns)

    # Set the basis shocks
    h_up, h_down = _set_sensi_shocks(rfs)
    h = h_up + h_down

    # Create diagonal shock matrices
    up_shocks = pd.DataFrame(np.diag(h_up), index=rfs, columns=rfs)
    down_shocks = pd.DataFrame(np.diag(-h_down), index=rfs, columns=rfs)

    # Calculate up/down instrument prices
    up_price = calc_prices(mkt_data, instr_info, ref_date, param_config,
                           der_underlyings, up_shocks)
    down_price = calc_prices(mkt_data, instr_info, ref_date, param_config,
                             der_underlyings, down_shocks)

    # Portfolio P&L change
    price_change = calc_portfolio_pnl(up_price - down_price).to_numpy().flatten()

    # Compute delta sensitivities (elementwise)
    deltas = price_change / h

    # Return as Pandas Series indexed by risk factors
    return pd.Series(deltas, index=rfs)

def _build_diag_shock_df(rfs: list[str], h: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        np.diag(h),
        index=rfs,
        columns=rfs
    )

def _calc_gamma_sensis(mkt_data: pd.DataFrame,
                       instr_info: pd.DataFrame,
                       ref_date: str,
                       param_config: dict,
                       der_underlyings: Dict[str, str]
                       ) -> pd.DataFrame:
    """Function calculating the gamma and cross-gamma sensitivities of a portfolio."""

    # Extract the risk factor names
    rfs = list(mkt_data.columns)

    # Set the basis shocks
    h_up, h_down = _set_sensi_shocks(rfs)
    h = h_up + h_down

    # Initialize matrix for (cross-)gamma sensitivities
    gamma = pd.DataFrame(0.0, index=rfs, columns=rfs)

    # Prebuild diagonal shocks
    h_mat = pd.DataFrame(np.diag(h), index=rfs, columns=rfs)

    # Loop through all risk factors
    for i, rf_i in enumerate(rfs):
        # Select a row of the diagonal shock matrix (series indexed by risk factors)
        row_i = h_mat.loc[rf_i]

        # Calculate all four shock combinations of up and down shocks
        shocks_up_up =  h_mat + row_i
        shocks_up_down =  h_mat - row_i
        shocks_down_up = -h_mat + row_i
        shocks_down_down = -h_mat - row_i

        # Calculate shocked portfolio values
        up_up_price = (calc_portfolio_pnl(calc_prices(mkt_data,
                                                      instr_info,
                                                      ref_date,
                                                      param_config,
                                                      der_underlyings,
                                                      shocks_up_up)).to_numpy().ravel())
        up_down_price = (calc_portfolio_pnl(calc_prices(mkt_data,
                                                        instr_info,
                                                        ref_date,
                                                        param_config,
                                                        der_underlyings,
                                                        shocks_up_down))
                                           .to_numpy().ravel())
        down_up_price = (calc_portfolio_pnl(calc_prices(mkt_data,
                                                        instr_info,
                                                        ref_date,
                                                        param_config,
                                                        der_underlyings,
                                                        shocks_down_up))
                                           .to_numpy().ravel())
        down_down_price = (calc_portfolio_pnl(calc_prices(mkt_data,
                                                          instr_info,
                                                          ref_date,
                                                          param_config,
                                                          der_underlyings,
                                                          shocks_down_down))
                                             .to_numpy().ravel())

        # Compute second order sensitivities for ith risk factor
        g_ij = (up_up_price - up_down_price - down_up_price + down_down_price) / (4.0 * h[i] * h)

        # Insert (cross-)gamma sensitivities for ith risk factor into matrix
        gamma.iloc[i, :] = g_ij
        gamma.iloc[:, i] = g_ij

    return gamma

def _get_greeks(mkt_data: pd.DataFrame,
                instr_info: pd.DataFrame,
                ref_date: str,
                param_config: dict,
                der_underlyings: Dict[str, str]
                ) -> pd.Series:
    """Function computing all first and second order portfolio sensitivities."""

    # Caculate first order portfolio sensitivities to the risk factors
    deltas = _calc_delta_sensis(mkt_data, instr_info, ref_date, param_config, der_underlyings)

    # Calculate second order portfolio sensitivities to the risk factors
    gammas = _calc_gamma_sensis(mkt_data, instr_info, ref_date, param_config, der_underlyings)

    return deltas, gammas

def calc_delta_scenario_pnl(mkt_data: pd.DataFrame,
                            instr_info: pd.DataFrame,
                            ref_date: str,
                            param_config: dict,
                            der_underlyings: Dict[str, str],
                            scenario_shocks: pd.DataFrame
                            ) -> pd.DataFrame:
    """Function calculating the scenario P&Ls approximated by Delta-Gamma method."""

    # Get the delta sensitivities of the portfolio to the risk factors
    deltas, _ = _get_greeks(mkt_data, instr_info, ref_date, param_config, der_underlyings)

    # Raise error if the inputs have mismatching risk factors
    factors = deltas.index
    if not factors.equals(scenario_shocks.columns):
        raise ValueError("Scenario shock columns do not match delta index")

    # Convert all inputs to numpy arrays
    delta = deltas.to_numpy()
    shocks = scenario_shocks.to_numpy()

    # Calculate the delta-approximated profit-and-losses
    pnl_delta = shocks @ delta

    return pd.DataFrame(pnl_delta, index=scenario_shocks.index, columns=["pnl"])

def calc_delta_gamma_scenario_pnl(mkt_data: pd.DataFrame,
                                  instr_info: pd.DataFrame,
                                  ref_date: str,
                                  param_config: dict,
                                  der_underlyings: Dict[str, str],
                                  scenario_shocks: pd.DataFrame
                                  ) -> pd.DataFrame:
    """Function calculating the scenario P&Ls approximated by Delta-Gamma method."""

    # Get the first and second order sensitivities of the portfolio to the risk factors
    deltas, gammas = _get_greeks(mkt_data, instr_info, ref_date, param_config, der_underlyings)

    # Raise error if any of the inputs have mismatching risk factors
    factors = deltas.index
    if not factors.equals(gammas.index):
        raise ValueError("Delta index and gamma index do not match")
    if not factors.equals(gammas.columns):
        raise ValueError("Gamma columns do not match delta index")
    if not factors.equals(scenario_shocks.columns):
        raise ValueError("Scenario shock columns do not match delta index")

    # Convert all inputs to numpy arrays
    delta = deltas.to_numpy()
    shocks = scenario_shocks.to_numpy()
    gamma = gammas.to_numpy()

    # Raise error if the gamma sensitivities matrix is not symmetric
    if not np.allclose(gamma, gamma.T):
        raise ValueError("Gamma matrix must be symmetric")

    # Calculate the first order profit-and-losses
    pnl_delta = shocks @ delta

    # Calculate the second order profit-and-losses
    pnl_gamma = 0.5 * np.sum(shocks * (shocks @ gamma), axis=1)

    # Sum up to get the total P&L
    pnl_total = pnl_delta + pnl_gamma

    return pd.DataFrame(pnl_total, index=scenario_shocks.index, columns=["pnl"])

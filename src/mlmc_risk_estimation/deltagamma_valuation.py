"""Module providing functions for Delta-Gamma approximate valuation
   of the risk factors."""

import numpy as np
import pandas as pd

from typing import List, Tuple

from full_valuation import calc_prices
from risk_aggregation import calc_portfolio_pnl

from typing import List, Tuple
import numpy as np
import pandas as pd

def _set_sensi_shocks(rfs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Function creating up and down shocks for each risk factor."""

    h_up = np.full(len(rfs), 0.001)
    h_down = np.full(len(rfs), 0.01) #h_up.copy()

    return h_up, h_down

def _calc_delta_sensis(mkt_data: pd.DataFrame,
                       instr_info: pd.DataFrame,
                       ref_date: str
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
    up_price = calc_prices(mkt_data, instr_info, ref_date, up_shocks)
    down_price = calc_prices(mkt_data, instr_info, ref_date, down_shocks)

    # Portfolio P&L change
    price_change = calc_portfolio_pnl(up_price - down_price).to_numpy().flatten()

    # Compute delta sensitivities (elementwise)
    deltas = price_change / h

    # Return as Pandas Series indexed by risk factors
    return pd.Series(deltas, index=rfs)

def _get_greeks(mkt_data: pd.DataFrame,
                instr_info: pd.DataFrame,
                ref_date: str
                ) -> pd.Series:
    """Function computing all first and second order instrument price sensitivities."""
    deltas = _calc_delta_sensis(mkt_data, instr_info, ref_date)
    return deltas

def calc_delta_scenario_pnl(mkt_data: pd.DataFrame,
                            instr_info: pd.DataFrame,
                            ref_date: str,
                            scenario_shocks: pd.DataFrame
                            ) -> pd.DataFrame:
    """Function calculating the scenario P&Ls approximated by Delta-Gamma method."""

    deltas = _get_greeks(mkt_data, instr_info, ref_date)

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

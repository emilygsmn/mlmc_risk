""""Module providing functions for risk aggregation."""

import pandas as pd
import numpy as np
import scipy as scp

__all__ = ["calc_instr_pnls", "calc_portfolio_pnl"]

def calc_instr_pnls(prices_at_t1: pd.DataFrame,
                    prices_at_t2: pd.DataFrame
                    ) -> pd.DataFrame:
    """Function calculating the scenario profit-and-loss per instrument."""

    # Ensure prices_at_t1 has exactly one row
    if prices_at_t1.shape[0] != 1:
        raise ValueError(f"prices_at_t1 must have exactly one row, got {prices_at_t1.shape[0]}")

    # Ensure the two DataFrames contain the same columns
    if not prices_at_t1.columns.equals(prices_at_t2.columns):
        raise ValueError("Column mismatch between prices_at_t1 and prices_at_t2")

    # Subtract the single row of prices_at_t1 from all rows in prices_at_t2
    return prices_at_t2.subtract(prices_at_t1.iloc[0])

def calc_portfolio_pnl(instr_pnls: pd.DataFrame
                       ) -> pd.DataFrame:
    """Function calculating the total portfolio scenario profit-and-loss."""

    # Ensure all values in the DataFrame are numeric
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in instr_pnls.dtypes):
        raise ValueError(f"Not all columns in the DataFrame are numeric: {instr_pnls}.")

    return instr_pnls.sum(axis=1).to_frame(name="total_pnl")

def apply_hd_weighting(vals, p):
    """Function applying Harrell-Davis weighting (assuming pre-sorted input vals).
       Source: scipy.stats.mstats.hdquantiles() documentation."""

    n = vals.size
    hd = np.empty((2), np.float64)
    if n < 2:
        hd.flat = np.nan
        return hd[0]
    v = np.arange(n+1) / float(n)
    betacdf = scp.stats.distributions.beta.cdf
    _w = betacdf(v, (n+1)*p, (n+1)*(1-p))
    w = _w[1:] - _w[:-1]
    hd_mean = np.dot(w, vals)
    hd[0] = hd_mean
    hd[1] = np.dot(w, (vals-hd_mean)**2)
    return hd[0]

def calc_standard_mc_hd_var(vals_df: pd.DataFrame,
                            conf_lvl: float
                            ) -> float:
    """Function calculating the Standard Monte Carlo Harrell-Davis VaR."""

    # Ensure correct input data type
    if not isinstance(vals_df, pd.DataFrame):
        raise TypeError("vals_df must be a pandas DataFrame.")

    # Ensure the DataFrame contains exactly one column
    if vals_df.shape[1] != 1:
        raise ValueError("vals_df must contain exactly one column.")

    # Ensure confidence level is in (0,1)
    if not 0 < conf_lvl < 1:
        raise ValueError("conf_lvl must be a numeric value strictly between 0 and 1.")

    # Ensure all data in the DataFrame is numeric
    col = vals_df.columns[0]
    if not pd.api.types.is_numeric_dtype(vals_df[col]):
        raise ValueError("The column in vals_df must be numeric.")

    # Extract numerical data from DataFrame (convert to numpy Array)
    vals_arr = vals_df[col].to_numpy(dtype=np.float64, copy=False)

    # Drop all NaNs if present
    vals_arr = vals_arr[~np.isnan(vals_arr)]

    # Return NaN if Array is empty
    if vals_arr.size == 0:
        return np.nan

    # Perform in-place numpy sorting (calculate order statistics)
    vals_arr.sort()

    # Calculate HD weighted average of order statistics
    hd_quantile = apply_hd_weighting(vals=vals_arr,
                                     p=1-conf_lvl)

    return abs(hd_quantile)

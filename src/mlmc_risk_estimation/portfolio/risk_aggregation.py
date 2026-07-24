"""Module providing functions for risk aggregation."""

import pandas as pd
import numpy as np
import scipy as scp

__all__ = ["calc_instr_pnls", "calc_portfolio_pnl", "calc_standard_mc_hd_var",
           "calc_tail_dependence_coeff"]

def calc_instr_pnls(prices_at_t1: pd.DataFrame,
                    prices_at_t2: pd.DataFrame
                    ) -> pd.DataFrame:
    """Calculate the scenario profit-and-loss per instrument."""

    if prices_at_t1.shape[0] != 1:
        raise ValueError(f"prices_at_t1 must have exactly one row, got {prices_at_t1.shape[0]}")

    if not prices_at_t1.columns.equals(prices_at_t2.columns):
        raise ValueError("Column mismatch between prices_at_t1 and prices_at_t2")

    return prices_at_t2.subtract(prices_at_t1.iloc[0])

def calc_portfolio_pnl(instr_pnls: pd.DataFrame,
                       weights: pd.Series | None = None
                       ) -> pd.DataFrame:
    """Calculate the total portfolio scenario profit-and-loss.

    If given, weights is a per-instrument (fin_instr) position factor (units held,
    scaled up to the benchmark's total of 1000) applied to the absolute price deltas in
    instr_pnls before summing across instruments.
    """

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in instr_pnls.dtypes):
        raise ValueError(f"Not all columns in the DataFrame are numeric: {instr_pnls}")

    if weights is not None:
        if not set(instr_pnls.columns).issubset(weights.index):
            raise ValueError("weights is missing entries for some instr_pnls columns")
        instr_pnls = instr_pnls.mul(weights.reindex(instr_pnls.columns), axis=1)

    return instr_pnls.sum(axis=1).to_frame(name="total_pnl")

def apply_hd_weighting(vals: np.ndarray, p: float) -> float:
    """Apply Harrell-Davis weighting (assuming pre-sorted input vals).
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
    """Calculate the Standard Monte Carlo Harrell-Davis VaR at level conf_lvl.

    Expects loss samples (positive = loss, negative = gain), s.t. the
    VaR is the conf_lvl-quantile (upper tail) of the loss."""

    if not isinstance(vals_df, pd.DataFrame):
        raise TypeError("vals_df must be a pandas DataFrame")

    if vals_df.shape[1] != 1:
        raise ValueError("vals_df must contain exactly one column")

    if not 0 < conf_lvl < 1:
        raise ValueError("conf_lvl must be a numeric value strictly between 0 and 1")

    col = vals_df.columns[0]
    if not pd.api.types.is_numeric_dtype(vals_df[col]):
        raise ValueError("The column in vals_df must be numeric")

    vals_arr = vals_df[col].to_numpy(dtype=np.float64, copy=False)

    # Drop NaNs before computing order statistics
    vals_arr = vals_arr[~np.isnan(vals_arr)]
    if vals_arr.size == 0:
        return np.nan

    vals_arr.sort()
    hd_quantile = apply_hd_weighting(vals=vals_arr,
                                     p=conf_lvl)

    return hd_quantile

def calc_tail_dependence_coeff(x: pd.Series,
                               y: pd.Series,
                               conf_lvl: float
                               ) -> float:
    """Calculate the empirical lower-tail dependence coefficient between x
       and y at conf_lvl"""

    if not 0 < conf_lvl < 1:
        raise ValueError("conf_lvl must be a numeric value strictly between 0 and 1")

    p = 1 - conf_lvl
    x_thresh = x.quantile(p)
    y_thresh = y.quantile(p)
    x_tail = x <= x_thresh

    if x_tail.sum() == 0:
        return np.nan

    return float((x_tail & (y <= y_thresh)).sum() / x_tail.sum())

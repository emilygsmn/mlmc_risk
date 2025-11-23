""""Module providing functions for calibration of the stochastic processes
    of the risk factors according to market data."""

import pandas as pd
import numpy as np

__all__ = ["calibrate_models"]

def _get_return_time_incr(ret_length):
    """Function getting the size of a time increment for return calculation."""

    if ret_length == "week":
        return 5
    elif ret_length == "quarter":
        return 13
    else:
        raise ValueError(f"Unknown return length '{ret_length}'.")

def _calc_rel_returns(time_series, ret_length):
    """Function calculating the relative returns given a series of prices."""

    # Calculate how large the time increments for the return calculation are
    time_incr_size = _get_return_time_incr(ret_length)

    # Select all relevant start and end prices for returns to be calculated
    starts = time_series[0 : len(time_series) - time_incr_size + 1 : time_incr_size]
    ends   = time_series[time_incr_size - 1 : len(time_series) : time_incr_size]

    # Get relative return of every non-overlapping period of days/months
    return (ends - starts) / starts

def _calc_abs_returns(time_series, ret_length):
    """Function calculating the absolute returns given a series of prices."""

    # Calculate how large the time increments for the return calculation are
    time_incr_size = _get_return_time_incr(ret_length)

    # Select all relevant start and end prices for returns to be calculated
    starts = time_series[0 : len(time_series) - time_incr_size + 1 : time_incr_size]
    ends   = time_series[time_incr_size - 1 : len(time_series) : time_incr_size]

    # Get absolute return of every non-overlapping period of days/months
    return ends - starts

def _calc_vola_from_time_series(returns, mu=0, dt=4):
    """Function calculating the empirical volatility from a time series."""

    rms_dev = np.sqrt(np.mean((returns - mu)**2))
    t_scaling = 1 / np.sqrt(dt)
    return rms_dev * t_scaling

def _get_num_returns_pa(ret_length):
    """Function getting the number of returns within a year, given a return period length."""

    if ret_length == "week":
        return 52
    elif ret_length == "quarter":
        return 4
    else:
        raise ValueError(f"Unknown return length '{ret_length}'.")

def _convert_to_rel_return_vola(emp_vola):
    """Fnction converting an absolute return volatility to a relative one."""
    return emp_vola

def _get_empirical_vola(time_series, ret_type="rel", ret_length="quarter", conv_to_rel=False):
    """Function getting the empirical volatility for a given time series."""

    # Calculate the returns by calling the respective return calculation function
    func_name = f"_calc_{ret_type}_returns"
    try:
        return_calc_func = globals()[func_name]
    except KeyError:
        raise RuntimeError(f"Return calculation function '{func_name}' not found.")
    returns = return_calc_func(time_series, ret_length)

    # Get the number of returns to be obtained per annum
    dt = _get_num_returns_pa(ret_length)

    # Calculate the empirical volatility from the return time series
    emp_vola = _calc_vola_from_time_series(returns=returns, mu=0, dt=dt)

    # If applicable, convert the abs. return vola to a rel. return vola
    if conv_to_rel:
        return _convert_to_rel_return_vola(emp_vola)
    else:
        return emp_vola

def calibrate_models(mkt_data, instr_info, param_config):
    """Function calibrating the stochastic processes to given market data."""

    calib_methods = param_config["valuation"]["calibr_methods"]
    rfs = mkt_data.columns

    instr_indexed = instr_info.set_index("fin_instr", drop=False)
    volas = pd.DataFrame(index=["sigma"], columns=rfs)

    for rf in rfs:
        risk_type = instr_indexed.loc[rf, "val_tag"]
        volas.loc["sigma", rf] = _get_empirical_vola(
            time_series=np.array(mkt_data[rf]),
            ret_type=calib_methods["return_type"][risk_type],
            ret_length=calib_methods["return_length"][risk_type],
            conv_to_rel=calib_methods["conv_to_rel"][risk_type]
            )

    return volas

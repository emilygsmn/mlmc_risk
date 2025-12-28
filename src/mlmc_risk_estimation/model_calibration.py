""""Module providing functions for calibration of the stochastic processes
    of the risk factors according to market data."""

import pandas as pd
import numpy as np

__all__ = ["calibrate_models"]

def _get_return_time_incr(ret_length):
    """Function getting the size of a time increment for return calculation."""

    # For weekly returns, one time increment consists of 5 days
    if ret_length == "week":
        return 5
    # For quarterly returns, one time increment consists of 13 weeks
    if ret_length == "quarter":
        return 13
    # Raise error in the case of another return length
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

    # Calculate the root mean square deviation of the returns from the mean
    rms_dev = np.sqrt(np.mean((returns - mu)**2))

    # Calculate time scaling factor
    t_scaling = 1 / np.sqrt(dt)

    # Apply time scaling to the root mean square deviation
    return rms_dev * t_scaling

def _get_num_returns_pa(ret_length):
    """Function getting the number of returns within a year, given a return period length."""

    # For weekly returns, there are 52 return values per year
    if ret_length == "week":
        return 52
    # For quarterly returns, there are 4 return values per year
    if ret_length == "quarter":
        return 4
    # Raise error in the case of another return length
    raise ValueError(f"Unknown return length '{ret_length}'.")

def _convert_to_rel_return_vola(emp_vola):
    """Function converting an absolute return volatility to a relative one."""
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
    return emp_vola

def calibrate_stoch_procs(mkt_data, instr_info, param_config):
    """Function calibrating the stochastic processes to given market data."""

    # Extract the calibration methods
    calib_methods = param_config["valuation"]["calibr_methods"]

    # Extract the risk factor names
    rfs = mkt_data.columns

    # Create indexed instrument info DataFrame for faster processing
    instr_indexed = instr_info.set_index("fin_instr", drop=False)

    # Initialize DataFrame for volatilities
    volas = pd.DataFrame(index=["sigma"], columns=rfs)

    # Loop through all risk factors
    for rf in rfs:
        # Extract the risk type of the current risk factor
        if rf.startswith("IR"):
            risk_type = "IR"
        else:
            risk_type = instr_indexed.loc[rf, "val_tag"]
        
        # Calculate the volatility for the chosen risk factor
        volas.loc["sigma", rf] = _get_empirical_vola(
            time_series=np.array(mkt_data[rf]),
            ret_type=calib_methods["return_type"][risk_type],
            ret_length=calib_methods["return_length"][risk_type],
            conv_to_rel=calib_methods["conv_to_rel"][risk_type]
            )

    return volas

def calc_set_credit_spreads(face_vals: pd.Series,
                            rfr: pd.Series,
                            maturities: pd.Series,
                            cra_bsp: pd.Series,
                            calib_targ: pd.Series
                            ) -> pd.Series:
    """Function calculating the set credit spreads using given calibration targets."""

    # Convert credit risk adjustment from basis points to decimal
    cra = cra_bsp / 10E+3

    # Calculate the implied rates given the calibration targets
    implied_rate = (face_vals.divide(calib_targ)).pow(1 / maturities)

    # Calculate set credit spreads
    set_cs = implied_rate - 1 - rfr - cra

    return set_cs

def calibrate_credit_spreads(instr_info:pd.DataFrame) -> pd.DataFrame:
    """Function calibrating the bond pricing model in terms of credit spreads."""

    # Select all instrument names for this val_tag
    mask = instr_info["val_tag"] == "ZCB_CS"
    instruments = instr_info.loc[mask, "fin_instr"].tolist()

    # Skip processing if no risk factor uses the current valuation type
    if not instruments:
        return instr_info

    # Determine which risk factors are needed
    rf_needed = instruments

    # Create indexed instrument info DataFrame for faster processing
    instr_indexed = instr_info.set_index("fin_instr", drop=False)

    fv = instr_indexed.loc[rf_needed, "notional_/_pos_units"]

    # Calculate the set credit spreads
    instr_indexed.loc[rf_needed, "set_cs"] = calc_set_credit_spreads(face_vals=fv.astype(float),
                            rfr=instr_indexed.loc[rf_needed, "rfr"].astype(float),
                            maturities=instr_indexed.loc[rf_needed, "maturity"].astype(float),
                            cra_bsp=instr_indexed.loc[rf_needed, "cra (bps)"].astype(float),
                            calib_targ=instr_indexed.loc[rf_needed, "calibration_target"].astype(float)
                            )

    return instr_indexed.reset_index(drop=True)

def calibrate_models(mkt_data, instr_info, param_config):

    # Volatility calibration for the stochastic processes
    volas = calibrate_stoch_procs(mkt_data, instr_info, param_config)

    # Set credit spread calibration
    instr_info = calibrate_credit_spreads(instr_info)

    return instr_info, volas

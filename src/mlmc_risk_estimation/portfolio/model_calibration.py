"""Module providing functions for calibration of the stochastic processes
    of the risk factors according to market data."""

import pandas as pd
import numpy as np

__all__ = ["calibrate_models"]

def _get_return_time_incr(ret_length: str) -> int:
    """Get the size of a time increment (in trading days) for return calculation."""

    # 1 week = 5 trading days; 1 quarter = 13 weeks = 65 trading days
    if ret_length == "week":
        return 5
    if ret_length == "quarter":
        return 65
    raise ValueError(f"Unknown return length '{ret_length}'")

def _calc_rel_returns(time_series: np.ndarray, ret_length: str) -> np.ndarray:
    """Calculate the relative returns given a series of prices."""

    incr = _get_return_time_incr(ret_length)

    # Non-overlapping incr-day periods, to avoid autocorrelation between consecutive returns
    points = time_series[::incr]
    return (points[1:] - points[:-1]) / points[:-1]

def _calc_abs_returns(time_series: np.ndarray, ret_length: str) -> np.ndarray:
    """Calculate the absolute returns given a series of prices."""

    incr = _get_return_time_incr(ret_length)

    # Non-overlapping incr-day periods, to avoid autocorrelation between consecutive returns
    points = time_series[::incr]
    return points[1:] - points[:-1]

def _calc_vola_from_time_series(returns: pd.Series, mu: float = 0, dt: int = 4) -> float:
    """Calculate the empirical volatility from a time series."""

    rms_dev = np.sqrt(np.mean((returns - mu)**2))

    # Annualize: dt is periods per year, so scale up by sqrt(periods per year)
    t_scaling = np.sqrt(dt)

    return rms_dev * t_scaling

def _get_num_returns_pa(ret_length: str) -> int:
    """Get the number of returns within a year, given a return period length."""

    if ret_length == "week":
        return 52
    if ret_length == "quarter":
        return 4
    raise ValueError(f"Unknown return length '{ret_length}'")

def _convert_to_rel_return_vola(emp_vola: float) -> float:
    """Convert an absolute return volatility to a relative one."""

    return emp_vola

def _get_empirical_vola(time_series: pd.Series,
                        ret_type: str = "rel",
                        ret_length: str = "quarter",
                        conv_to_rel: bool = False
                        ) -> float:
    """Get the empirical volatility for a given time series."""

    func_name = f"_calc_{ret_type}_returns"
    try:
        return_calc_func = globals()[func_name]
    except KeyError:
        raise RuntimeError(f"Return calculation function '{func_name}' not found")
    returns = return_calc_func(time_series, ret_length)

    dt = _get_num_returns_pa(ret_length)
    emp_vola = _calc_vola_from_time_series(returns=returns, mu=0, dt=dt)

    if conv_to_rel:
        return _convert_to_rel_return_vola(emp_vola)
    return emp_vola

def _calc_smoothed_ir_vola(time_series: np.ndarray,
                           shift: float,
                           trd_days_per_week: int = 5,
                           weeks_per_quarter: int = 13,
                           quarters_pa: int = 4
                           ) -> float:
    """Calculate the rolling-window smoothed volatility for interest rate factors.

    Averages the annualized empirical standard deviation of shifted-log quarterly (13-week)
    returns over all 13 weekly starting offsets. The shift keeps the log defined for the
    negative rates that a plain relative return cannot handle near zero.
    """

    log_shifted = np.log(time_series + shift)
    step = weeks_per_quarter * trd_days_per_week
    scaling = np.sqrt(quarters_pa)

    offset_volas = []
    for offset_weeks in range(weeks_per_quarter):
        # Non-overlapping 13-week returns, each offset starting one week later
        points = log_shifted[offset_weeks * trd_days_per_week :: step]
        returns = points[1:] - points[:-1]
        if returns.size < 2:
            continue
        offset_volas.append(scaling * np.std(returns, ddof=1))

    return float(np.mean(offset_volas))

def calibrate_stoch_procs(mkt_data: pd.DataFrame,
                          instr_info: pd.DataFrame,
                          param_config: dict
                          ) -> pd.DataFrame:
    """Calibrate the stochastic processes to given market data."""

    calib_methods = param_config["valuation"]["calibr_methods"]
    sgbm_shift = param_config["valuation"]["sgbm_shift"]
    rfs = mkt_data.columns
    instr_indexed = instr_info.set_index("fin_instr", drop=False)
    volas = pd.DataFrame(index=["sigma"], columns=rfs)

    for rf in rfs:
        time_series = np.asarray(mkt_data[rf], dtype=float)

        if rf.startswith("IR"):
            # Shifted-log rolling-window estimator
            volas.loc["sigma", rf] = _calc_smoothed_ir_vola(time_series, shift=sgbm_shift)
        else:
            risk_type = "EQVOL" if rf.startswith("EQVOL") else instr_indexed.loc[rf, "val_tag"]
            volas.loc["sigma", rf] = _get_empirical_vola(
                time_series=time_series,
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
    """Calculate the set credit spreads using given calibration targets."""

    cra = cra_bsp / 10E+3
    implied_rate = (face_vals.divide(calib_targ)).pow(1 / maturities)
    set_cs = implied_rate - 1 - rfr - cra

    return set_cs

def calibrate_credit_spreads(instr_info: pd.DataFrame) -> pd.DataFrame:
    """Calibrate the bond pricing model in terms of credit spreads."""

    mask = instr_info["val_tag"] == "ZCB_CS"
    rf_needed = instr_info.loc[mask, "fin_instr"].tolist()

    if not rf_needed:
        return instr_info

    instr_indexed = instr_info.set_index("fin_instr", drop=False)

    instr_indexed.loc[rf_needed, "set_cs"] = calc_set_credit_spreads(
        face_vals=instr_indexed.loc[rf_needed, "face_value"].astype(float),
        rfr=instr_indexed.loc[rf_needed, "rfr"].astype(float),
        maturities=instr_indexed.loc[rf_needed, "maturity"].astype(float),
        cra_bsp=instr_indexed.loc[rf_needed, "cra (bps)"].astype(float),
        calib_targ=instr_indexed.loc[rf_needed, "calibration_target"].astype(float)
        )

    return instr_indexed.reset_index(drop=True)

def calc_set_inflation(face_vals: pd.Series,
                       rfr: pd.Series,
                       maturities: pd.Series,
                       calib_targ: pd.Series
                       ) -> pd.Series:
    """Calculate the set inflation using given calibration targets."""

    inv_disc_fact = (1 + rfr).pow(maturities)
    implied_rate = (calib_targ.multiply(inv_disc_fact).divide(face_vals))
    set_infl = implied_rate.pow(1 / maturities) - 1

    return set_infl

def calibrate_inflation(instr_info: pd.DataFrame) -> pd.DataFrame:
    """Calibrate the bond pricing model in terms of inflation."""

    mask = instr_info["val_tag"] == "ZCB_INFL"
    rf_needed = instr_info.loc[mask, "fin_instr"].tolist()

    if not rf_needed:
        return instr_info

    instr_indexed = instr_info.set_index("fin_instr", drop=False)

    instr_indexed.loc[rf_needed, "set_infl"] = calc_set_inflation(
        face_vals=instr_indexed.loc[rf_needed, "face_value"].astype(float),
        rfr=instr_indexed.loc[rf_needed, "rfr"].astype(float),
        maturities=instr_indexed.loc[rf_needed, "maturity"].astype(float),
        calib_targ=instr_indexed.loc[rf_needed, "calibration_target"].astype(float)
        )

    return instr_indexed.reset_index(drop=True)

def calibrate_models(mkt_data: pd.DataFrame,
                     instr_info: pd.DataFrame,
                     param_config: dict
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate all pricing models to historical data."""

    volas = calibrate_stoch_procs(mkt_data, instr_info, param_config)
    instr_info = calibrate_credit_spreads(instr_info)
    instr_info = calibrate_inflation(instr_info)

    return instr_info, volas

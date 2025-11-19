""""Module providing functions for full valuation of the risk factors."""

import numpy as np

__all__ = ["comp_prices_with_calib_targets"]

def _get_market_price(market_data, instr_name, ref_date):
    """Function selecting the relevant market price from the historical time series'."""
    return market_data.loc[market_data["Date"] == ref_date, instr_name].iloc[0]

def _calc_FX_price(market_data, instr_name, ref_date):
    """Function getting the FX rate quoted in EUR as of the reference date."""
    return _get_market_price(market_data, instr_name, ref_date)

def _calc_EQ_price(market_data, instr_name, ref_date):
    """Function getting the equity market price as of the reference date."""
    return _get_market_price(market_data, instr_name, ref_date)

def _calc_BOND_price(instr_info):
    """Function pricing a zero-coupon bond excl. inflation and credit risk."""
    face_val = instr_info["notional_/_pos_units"]
    rfr = instr_info["rfr"]
    maturity = instr_info["maturity"]
    return face_val * np.exp(-rfr * maturity)

def comp_prices_with_calib_targets(base_values, calib_target):
    """Function checking whether the base values are close enough to the calibration targets."""

    # Compute the absolute value of the difference between each price and the target value
    eps = 10E-5
    exceeds = (base_values["price"] - calib_target["calib_target"]).abs() > eps

    # Print the result of the price comparison
    if exceeds.any():
        print("The base prices of the following instruments deviate strongly from the"
              "EIOPA calibration targets:")
        print(base_values.loc[exceeds, "fin_instr"].tolist())
    else:
        print("All base prices are close to the EIOPA calibration targets.")

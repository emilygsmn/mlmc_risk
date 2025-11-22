""""Module providing functions for full valuation of the risk factors."""

import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

__all__ = ["comp_prices_with_calib_targets"]

def _get_pricing_func(tag: str):
    """ Function returning the pricing function object for a given tag, using the naming convention.
    Raises NotImplementedError if not found.
    """
    module = sys.modules[__name__]
    func_name = f"_calc_{tag}_price"
    func = getattr(module, func_name, None)
    if func is None:
        raise NotImplementedError(f"No pricing function found for val_tag='{tag}'")
    return func

def _get_mtm_base_value(mkt_data, rfs, ref_date):
    """Function selecting the relevant market price from the historical time series'."""
    return mkt_data.loc[[ref_date], rfs]

def _calc_FX_price(rfs, mkt_data, shocks, ref_date):
    """Function getting the FX rate quoted in EUR as of the reference date."""

    base_df = _get_mtm_base_value(mkt_data, rfs, ref_date)
    if base_df.shape[0] != 1:
        raise ValueError(f"_get_mtm_base_value must return exactly one row for ref_date={ref_date}")

    base_series = base_df.iloc[0]

    priced = (1 + shocks).multiply(base_series, axis=1)

    return priced

def _calc_EQ_price(rfs, mkt_data, shocks, ref_date):
    """Function getting the equity market price as of the reference date."""

    base_df = _get_mtm_base_value(mkt_data, rfs, ref_date)
    if base_df.shape[0] != 1:
        raise ValueError(f"_get_mtm_base_value must return exactly one row for ref_date={ref_date}")

    base_series = base_df.iloc[0]

    priced = (1 + shocks).multiply(base_series, axis=1)

    return priced

def _calc_BOND_price(instr_info, shocks):
    """Function pricing a zero-coupon bond excl. inflation and credit risk."""
    face_val = instr_info["notional_/_pos_units"]
    rfr = instr_info["rfr"]
    maturity = instr_info["maturity"]
    return face_val * np.exp(-(rfr + shocks) * maturity)

def _convert_loc_ccy_to_eur(prices_loc, instr_info):
    """Function converting prices quoted in local currency to EUR values."""
    prices_eur = prices_loc.copy()

    for _, row in instr_info.iterrows():
        instr = row["fin_instr"]
        ccy = row["ccy"]

        # Skip EUR-denominated instruments
        if instr.startswith("FX-") or ccy == "EUR":
            continue

        # Construct FX column name
        fx_col_candidates = [col for col in prices_loc.columns if col.startswith(f"FX-{ccy}-")]
        if not fx_col_candidates:
            raise KeyError(f"No FX column found for currency '{ccy}' required for '{instr}'")
        fx_col = fx_col_candidates[0]

        # Multiply the instrument prices by the FX column (broadcast across rows)
        prices_eur[instr] = prices_loc[instr] * prices_loc[fx_col]

    return prices_eur

def calc_prices(mkt_data, params, instr_info, ref_date, shocks=None):
    """Function running the pricing functions for all financial instruments grouped by val_tag."""

    # Set the argument specifications
    arg_spec: Dict[str, Tuple[Any, ...]] = {
        "FX": ("rfs", "mkt_data", "shocks", "ref_date"),
        "INFL": ("rfs", "mkt_data", "shocks", "ref_date"),
        "EQ": ("rfs", "mkt_data", "shocks", "ref_date"),
        "BOND": ("rf_rates", "notional", "tenor", "shocks")
        }

    # Set shock to zero for base scenario valuation
    if shocks is None:
        shocks = pd.DataFrame(
            data=[np.zeros(len(instr_info["fin_instr"]))],
            columns=instr_info["fin_instr"]
            )

    # Initialize list to collect the data for different valuation types in
    results = []

    # Loop by valuation type
    for val_tag in instr_info["val_tag"].unique():

        # Select all instrument names for this val_tag
        mask = instr_info["val_tag"] == val_tag
        instruments = instr_info.loc[mask, "fin_instr"].tolist()

        # Skip processing if no risk factor uses the current valuation type
        if not instruments:
            continue

        # Determine which risk factors are needed
        rf_needed = instruments

        # Load the correct pricing function by naming convention
        func_name = f"_calc_{val_tag}_price"
        try:
            price_func = globals()[func_name]
        except KeyError:
            raise RuntimeError(f"Pricing function '{func_name}' not found.")

        # Lookup argument order in ARG_SPEC and select only relevant risk factor data
        arg_list = []
        for arg in arg_spec[val_tag]:
            if arg == "rfs":
                arg_list.append(rf_needed)
            elif arg == "shocks":
                shocks_sub = shocks[rf_needed]
                arg_list.append(shocks_sub)
            elif arg == "mkt_data":
                mkt_sub    = mkt_data[rf_needed]
                arg_list.append(mkt_sub)
            elif arg == "params":
                params_sub = params[rf_needed]
                arg_list.append(params_sub)
            elif arg == "ref_date":
                arg_list.append(ref_date)
            else:
                raise ValueError(f"Unknown argument specifier '{arg}'.")

        # Call the relevant pricing function
        prices = price_func(*arg_list)

        # priced_df must contain exactly the same columns requested
        prices = prices[rf_needed]

        # Collect the resulting data frames
        results.append(prices)

    # Combine all valuation outputs into one data frame
    if results:
        final = pd.concat(results, axis=1)
    else:
        final = pd.DataFrame()

    # Convert all local currency prices to EUR
    final = _convert_loc_ccy_to_eur(final, instr_info)

    return final

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

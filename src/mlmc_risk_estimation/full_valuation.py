""""Module providing functions for full valuation of the risk factors."""

import sys

import numpy as np
import pandas as pd

from utils.introspection import get_pricing_arg_spec, get_pricing_func

__all__ = ["calc_prices", "comp_prices_with_calib_targets"]

def _get_mtm_base_value(mkt_data: pd.DataFrame,
                        rfs: list,
                        ref_date: str
                        ) -> pd.DataFrame:
    """Function selecting the relevant market price from the historical time series'."""
    return mkt_data.loc[[ref_date], rfs]

def _calc_FX_price(rfs: list,
                   mkt_data: pd.DataFrame,
                   shocks: pd.DataFrame,
                   ref_date: str
                   ) -> pd.DataFrame:
    """Function getting the FX rate quoted in EUR as of the reference date."""

    # Extract market value at reference date
    base_df = _get_mtm_base_value(mkt_data, rfs, ref_date)

    # Ensure the DataFrame contains exactly one row (market values of at the one chosen date)
    if base_df.shape[0] != 1:
        raise ValueError(f"_get_mtm_base_value must return exactly one row for ref_date={ref_date}")

    # Convert the row data to a Series
    base_series = base_df.iloc[0]

    # Apply the (multiplicative) FX shocks to the base values
    priced = (1 + shocks).multiply(base_series, axis=1)

    return priced

def _calc_EQ_price(rfs: list,
                   mkt_data: pd.DataFrame,
                   shocks: pd.DataFrame,
                   ref_date: str
                   ) -> pd.DataFrame:
    """Function getting the equity market price as of the reference date."""

    # Extract market value at reference date
    base_df = _get_mtm_base_value(mkt_data, rfs, ref_date)

    # Ensure the DataFrame contains exactly one row (market values of at the one chosen date)
    if base_df.shape[0] != 1:
        raise ValueError(f"_get_mtm_base_value must return exactly one row for ref_date={ref_date}")

    # Convert the row data to a Series
    base_series = base_df.iloc[0]

    # Apply the (multiplicative) equity shocks to the base values
    priced = (1 + shocks).multiply(base_series, axis=1)

    return priced

def _calc_ZCB_price(face_vals: pd.Series,
                     riskfree_rates: pd.Series,
                     maturities: pd.Series,
                     shocks: pd.DataFrame,
                     ) -> pd.DataFrame:
    """Function pricing a zero-coupon bond excl. inflation and credit risk."""

    # Apply the (additive) interest rate shocks to the base rates
    shocked_rfr = shocks.add(riskfree_rates, axis=1)

    # Calculate the discount factors based on the shocked rates (continuous compounding)
    exponent = - shocked_rfr.multiply(maturities, axis=1)
    disc_fact = np.exp(exponent)

    # Calculate present values of the face values
    prices = face_vals.multiply(disc_fact, axis=1)

    return prices

def _calc_ZCB_INFL_price(face_vals: pd.Series,
                         riskfree_rates: pd.Series,
                         maturities: pd.Series,
                         set_infl: pd.Series,
                         shocks: pd.DataFrame
                         ) -> pd.DataFrame:
    """Function pricing an inflation-linked zero-coupon bond excl. credit risk."""

    # Apply inflation to the face values
    infl_fact = (1 + set_infl).pow(maturities)
    infl_adj_face_vals = face_vals.multiply(infl_fact)

    # Apply the (additive) interest rate shocks to the base rates
    shocked_rfr = shocks.add(riskfree_rates, axis=1)

    # Calculate the discount factors based on the shocked rates (discrete compounding)
    disc_fact = 1 / (1 + shocked_rfr).pow(maturities)

    # Calculate present values of the face values
    prices = infl_adj_face_vals.multiply(disc_fact, axis=1)

    return prices

def _calc_ZCB_CS_price(face_vals: pd.Series,
                       riskfree_rates: pd.Series,
                       maturities: pd.Series,
                       cra_bsp: pd.Series,
                       set_cs: pd.Series,
                       shocks: pd.DataFrame
                       ) -> pd.DataFrame:
    """Function pricing a zero-coupon bond with credit risk, excl. inflation."""

    # Apply the (additive) interest rate shocks to the base rates
    shocked_rfr = shocks.add(riskfree_rates, axis=1)

    # Convert credit risk adjustment from basis points to decimal
    cra = cra_bsp / 10E+3

    # Calculate the discount factors based on the shocked rates (discrete compounding)
    disc_fact = 1 / (1 + shocked_rfr + cra + set_cs).pow(maturities)

    # Calculate present values of the face values
    prices = face_vals.multiply(disc_fact, axis=1)

    return prices

def _convert_loc_ccy_to_eur(prices_loc: pd.DataFrame,
                            instr_info: pd.DataFrame
                            ) -> pd.DataFrame:
    """Function converting prices quoted in local currency to EUR values."""

    # Create copy of DataFrame for out-of-place modification
    prices_eur = prices_loc.copy()

    for _, row in instr_info.iterrows():

        # Select current instrument and currency
        instr = row["fin_instr"]
        val_tag = row["val_tag"]
        ccy = row["ccy"]

        # Skip EUR-denominated instruments
        if val_tag == "FX" or ccy == "EUR":
            continue

        # Construct FX column name
        fx_col_candidates = [col for col in prices_loc.columns if col.startswith(f"FX-{ccy}-")]
        if not fx_col_candidates:
            raise KeyError(f"No FX column found for currency '{ccy}' required for '{instr}'")
        fx_col = fx_col_candidates[0]

        # Multiply the instrument prices by the FX column (broadcast across rows)
        prices_eur[instr] = prices_loc[instr] * prices_loc[fx_col]

    return prices_eur

def _build_rf_shock_df(rf_needed: list,
                       instr_indexed: pd.DataFrame,
                       shocks: pd.DataFrame,
                       mat_col: str = 'maturity',
                       shocks_prefix: str = 'IR_EUR_'
                       ) -> pd.DataFrame:
    """Builds a DataFrame with one column per element in rf_needed."""

    cols = {}
    for rf in rf_needed:
        # Ensure rf exists in instr_indexed index
        if rf not in instr_indexed.index:
            raise KeyError(f"Risk factor '{rf}' not found in instr_indexed.index")
        mat = instr_indexed.at[rf, mat_col]

        # Format maturity to 2-digit string if numeric-like
        try:
            mat_str = f"{int(mat):02d}"
        except Exception:
            mat_str = str(mat)

        shocks_col = f"{shocks_prefix}{mat_str}"
        if shocks_col not in shocks.columns:
            raise KeyError(f"Column '{shocks_col}' not found in shocks")

        # Take the series from shocks (alignment by index will happen automatically)
        cols[rf] = shocks[shocks_col]

    # Build DataFrame from dict of Series (preserves shocks index / aligns indexes)
    return pd.DataFrame(cols)

def calc_prices(mkt_data: pd.DataFrame,
                instr_info: pd.DataFrame,
                ref_date: str,
                shocks: pd.DataFrame | None = None
                ) -> pd.DataFrame:
    """Function running the pricing functions for all financial instruments grouped by val_tag."""

    # Get the dict of all pricing argument specs
    arg_spec = get_pricing_arg_spec(module=sys.modules[__name__],
                                     prefix="_calc_",
                                     suffix="_price")

    # Set shock to zero for base scenario valuation
    if shocks is None:
        shocks = pd.DataFrame(
            data=[np.zeros(mkt_data.shape[1])],
            columns=mkt_data.columns
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
        price_func = get_pricing_func(tag=val_tag,
                                      module=sys.modules[__name__]
                                      )

        # Lookup argument order in arg_spec and select only relevant risk factor data
        instr_indexed = instr_info.set_index("fin_instr", drop=False)
        arg_list = []
        for arg in arg_spec[val_tag]:
            if arg == "rfs":
                arg_list.append(rf_needed)
            elif arg == "mkt_data":
                mkt_sub = mkt_data[rf_needed]
                arg_list.append(mkt_sub)
            elif arg == "riskfree_rates":
                rfr_sub = instr_indexed.loc[rf_needed, "rfr"]
                arg_list.append(rfr_sub.astype(float))
            elif arg == "face_vals":
                face_vals = instr_indexed.loc[rf_needed, "notional_/_pos_units"]
                arg_list.append(face_vals.astype(float))
            elif arg == "maturities":
                maturities = instr_indexed.loc[rf_needed, "maturity"]
                arg_list.append(maturities.astype(float))
            elif arg == "cra_bps":
                cra_bps = instr_indexed.loc[rf_needed, "cra (bps)"]
                arg_list.append(cra_bps.astype(float))
            elif arg == "shocks":
                if val_tag == "ZCB":
                    shocks_sub = _build_rf_shock_df(rf_needed, instr_indexed, shocks,
                      mat_col='maturity', shocks_prefix='IR_EUR_')
                else:
                    shocks_sub = shocks[rf_needed]
                arg_list.append(shocks_sub)
            elif arg == "ref_date":
                arg_list.append(ref_date)
            else:
                raise ValueError(f"Unknown argument specifier '{arg}'.")

        # Call the relevant pricing function
        prices = price_func(*arg_list)

        # prices must contain exactly the same columns requested
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

def comp_prices_with_calib_targets(base_values: pd.DataFrame,
                                   calib_target: pd.DataFrame
                                   ):
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

""""Module providing functions for full valuation of the risk factors."""

import sys
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm
from numpy.typing import NDArray

from utils.introspection import get_pricing_arg_spec, get_pricing_func

__all__ = ["calc_prices", "comp_prices_with_calib_targets"]

def _get_mtm_base_value(mkt_data: pd.DataFrame,
                        ref_date: str,
                        rfs: list | None = None
                        ) -> NDArray[np.floating]:
    """Function selecting the relevant market price from the historical time series'."""
    if rfs is None:
        return mkt_data.loc[[ref_date]]
    return mkt_data.loc[[ref_date], rfs]

def _apply_rf_shocks(base_rf_vals: pd.DataFrame,
                     shocks: pd.DataFrame,
                     shock_types: Dict[str, str]
                     ) -> pd.DataFrame:
    """Function applying all risk factor shocks to the base values."""

    # Broadcast base values to scenario dimension
    base = pd.concat([base_rf_vals] * len(shocks))

    # Align indices
    base.index = shocks.index

    # Initialize DataFrame for shocked risk factor values
    shocked = base.copy()

    # Loop through all tisk types
    for risk_type, appl_method in shock_types.items():
        # Extract the risk factors belonging to current risk type
        cols = [c for c in shocked.columns if risk_type in c]
        if appl_method == "add":
            # Apply additive shocks
            shocked[cols] = base[cols] + shocks[cols]
        elif appl_method == "mult":
            # Apply multiplicative shocks
            shocked[cols] = base[cols] * (1 + shocks[cols])

    return shocked

def _build_rf_shock_df(rf_needed: list,
                       instr_indexed: pd.DataFrame,
                       shocked_rf_vals: pd.DataFrame
                       ) -> pd.DataFrame:
    """Builds a DataFrame with one column per element in rf_needed."""

    # Initialize dictionary of data columns
    cols = {}

    # Loop through all relevant risk factors
    for rf in rf_needed:

        # Ensure rf exists in instr_indexed index
        if rf not in instr_indexed.index:
            raise KeyError(f"Risk factor '{rf}' not found in instr_indexed.index")

        # Get maturity and currency of the selected risk factor
        mat = instr_indexed.at[rf, "maturity"]
        ccy = instr_indexed.at[rf, "ccy"]

        # Format maturity to 2-digit string if numeric-like
        try:
            mat_str = f"{int(mat):02d}"
        except Exception:
            mat_str = str(mat)
        shocks_col = f"IR_{ccy}_{mat_str}"
        if shocks_col not in shocked_rf_vals.columns:
            raise KeyError(f"Column '{shocks_col}' not found in shocks")

        # Take the series from shocks (alignment by index will happen automatically)
        cols[rf] = shocked_rf_vals[shocks_col]

    # Build DataFrame from dict of Series (preserves shocks index / aligns indexes)
    return pd.DataFrame(cols)

def _calc_FX_price(mkt_rates: NDArray[np.floating],
                   ) -> NDArray[np.floating]:
    """Function getting the FX rate quoted in EUR as of the reference date."""

    return mkt_rates

def _calc_EQ_price(mkt_rates: NDArray[np.floating]
                   ) -> NDArray[np.floating]:
    """Function getting the equity market price as of the reference date."""

    return mkt_rates

def _calc_ZCB_price(face_vals: NDArray[np.floating],
                     maturities: NDArray[np.floating],
                     riskfree_rates: NDArray[np.floating]
                     ) -> NDArray[np.floating]:
    """Function pricing a zero-coupon bond excl. inflation and credit risk."""

    # Calculate the discount factors based on the shocked rates (continuous compounding)
    exponent = - riskfree_rates * maturities
    disc_fact = np.exp(exponent)

    # Calculate present values of the face values
    prices = disc_fact * face_vals

    return prices

def _calc_ZCB_INFL_price(face_vals: NDArray[np.floating],
                         maturities: NDArray[np.floating],
                         riskfree_rates: NDArray[np.floating],
                         set_infl: NDArray[np.floating]
                         ) -> NDArray[np.floating]:
    """Function pricing an inflation-linked zero-coupon bond excl. credit risk."""

    # Apply inflation to the face values
    infl_fact = (1 + set_infl) ** maturities
    infl_adj_face_vals = face_vals * infl_fact

    # Calculate the discount factors based on the shocked rates (discrete compounding)
    disc_fact = 1 / (1 + riskfree_rates) ** maturities

    # Calculate present values of the face values
    prices = disc_fact * infl_adj_face_vals

    return prices

def _calc_ZCB_CS_price(face_vals: NDArray[np.floating],
                       maturities: NDArray[np.floating],
                       riskfree_rates: NDArray[np.floating],
                       cra_bps: NDArray[np.floating],
                       set_cs: NDArray[np.floating]
                       ) -> NDArray[np.floating]:
    """Function pricing a zero-coupon bond with credit risk, excl. inflation."""

    # Convert credit risk adjustment from basis points to decimal
    cra = cra_bps / 10E+3

    # Calculate the discount factors based on the shocked rates (discrete compounding)
    disc_fact = 1 / (1 + riskfree_rates + cra + set_cs) ** maturities

    # Calculate present values of the face values
    prices = disc_fact * face_vals

    return prices

def _calc_PUT_price(
    spots: NDArray[np.floating],   # shape (n, k)
    strikes: NDArray[np.floating], # shape (k,)
    maturities: NDArray[np.floating],    # shape (k,)
    riskfree_rates: NDArray[np.floating],    # shape (n, k)
    volas: NDArray[np.floating],   # shape (n, k)
) -> NDArray[np.floating]:
    """Function calculating Black–Scholes prices for European put options."""

    # Compute time scaling factor
    time_fact = np.sqrt(maturities)

    # Calculate parameters for Black-Scholes pricing function
    d1 = (
        np.log(spots / strikes)
        + (riskfree_rates + 0.5 * volas**2) * maturities
    ) / (volas * time_fact)
    d2 = d1 - volas * time_fact

    # Calculate European put option prices
    prices = (
        strikes * np.exp(-riskfree_rates * maturities) * norm.cdf(-d2)
        - spots * norm.cdf(-d1)
    )

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

def calc_prices(mkt_data: pd.DataFrame,
                instr_info: pd.DataFrame,
                ref_date: str,
                param_config: Dict[str, str],
                der_underlyings: Dict[str, str],
                shocks: pd.DataFrame | None = None
                ) -> pd.DataFrame:
    """Function running the pricing functions for all financial instruments grouped by val_tag."""

    # Get the dict of all pricing argument specs
    arg_spec = get_pricing_arg_spec(module=sys.modules[__name__],
                                     prefix="_calc_",
                                     suffix="_price")

    # Apply the risk factor shocks to the base rates if scenario shocks are given
    is_base_scenario = shocks is None
    base_rf_vals = _get_mtm_base_value(mkt_data, ref_date)
    if not is_base_scenario:
        shock_types = param_config["valuation"]["shock_type"]
        shocked_rf_vals = _apply_rf_shocks(base_rf_vals, shocks, shock_types)
        # Initialize empty DataFrame for results
        final = pd.DataFrame(index=shocks.index)
    else:
        shocked_rf_vals = base_rf_vals
        # Initialize empty DataFrame for results
        final = pd.DataFrame(index=base_rf_vals.index)

    # Loop by valuation type
    val_tags = instr_info["val_tag"].unique()
    for val_tag in val_tags:

        # Select all instrument names for this val_tag
        mask = instr_info["val_tag"] == val_tag
        rf_needed = instr_info.loc[mask, "fin_instr"].tolist()

        # Skip processing if no risk factor uses the current valuation type
        if not rf_needed:
            continue

        # Load the correct pricing function by naming convention
        price_func = get_pricing_func(tag=val_tag,
                                      module=sys.modules[__name__]
                                      )

        # Create indexed instrument info DataFrame for faster processing
        instr_indexed = instr_info.set_index("fin_instr", drop=False)

        def arg_source(arg_name: str, shocked_rf_vals: pd.DataFrame):

            if arg_name == "mkt_rates":
                if not is_base_scenario:
                    return shocked_rf_vals[rf_needed]
                else:
                    return _get_mtm_base_value(mkt_data, ref_date, rf_needed).to_numpy()

            elif arg_name == "spots":
                underlying_cols = [der_underlyings[d] for d in rf_needed]
                spots_data = final.loc[:, underlying_cols].to_numpy()
                return spots_data

            elif arg_name == "strikes":
                # derive strikes from spots for the same valuation tag
                underlying_cols = [der_underlyings[d] for d in rf_needed]
                strikes_data = base_rf_vals[underlying_cols].to_numpy()
                return strikes_data[0]

            elif arg_name == "maturities":
                return instr_indexed.loc[rf_needed, "maturity"].to_numpy(dtype=float)

            elif arg_name == "riskfree_rates":
                return _build_rf_shock_df(rf_needed=rf_needed, 
                                              instr_indexed=instr_indexed,
                                              shocked_rf_vals=shocked_rf_vals
                                              ).to_numpy()

            elif arg_name == "volas":
                underlying_cols = [der_underlyings[d] for d in rf_needed]
                spots_data = final.loc[:, underlying_cols].to_numpy()
                return spots_data / 5000

            elif arg_name == "face_vals":
                return instr_indexed.loc[rf_needed, "notional_/_pos_units"].to_numpy(dtype=float)

            elif arg_name == "cra_bps":
                return instr_indexed.loc[rf_needed, "cra (bps)"].to_numpy(dtype=float)

            elif arg_name == "set_cs":
                return instr_indexed.loc[rf_needed, "set_cs"].to_numpy(dtype=float)

            elif arg_name == "set_infl":
                return instr_indexed.loc[rf_needed, "set_infl"].to_numpy(dtype=float)

            elif arg_name == "ref_date":
                return ref_date

            else:
                raise ValueError(f"Unknown argument specifier '{arg_name}'.")

        arg_values = [arg_source(arg, shocked_rf_vals) for arg in arg_spec[val_tag]]

        # Call the relevant pricing function
        prices = price_func(*arg_values)

        # Assign prices directly into the final DataFrame
        final.loc[:, rf_needed] = prices

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

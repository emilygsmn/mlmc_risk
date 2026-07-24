"""Module providing functions for full valuation of the risk factors."""

import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from numpy.typing import NDArray

from mlmc_risk_estimation.portfolio.utils.introspection import (
    get_pricing_arg_spec,
    get_pricing_func
)

__all__ = ["calc_prices", "comp_prices_with_calib_targets"]

def _get_mtm_base_value(mkt_data: pd.DataFrame,
                        ref_date: str,
                        rfs: list[str] | None = None
                        ) -> NDArray[np.floating]:
    """Select the relevant market price from the historical time series'."""

    if rfs is None:
        return mkt_data.loc[[ref_date]]
    
    return mkt_data.loc[[ref_date], rfs]

def _apply_rf_shocks(base_rf_vals: pd.DataFrame,
                     shocks: pd.DataFrame,
                     shock_types: dict[str, str]
                     ) -> pd.DataFrame:
    """Apply all risk factor shocks to the base values."""

    # Broadcast base values to scenario dimension
    base = pd.concat([base_rf_vals] * len(shocks))
    base.index = shocks.index
    shocked = base.copy()

    for risk_type, appl_method in shock_types.items():
        # Risk factors are matched by risk_type appearing anywhere in the column name
        cols = [c for c in shocked.columns if risk_type in c]
        if appl_method == "add":
            shocked[cols] = base[cols] + shocks[cols]
        elif appl_method == "mult":
            shocked[cols] = base[cols] * (1 + shocks[cols])

    return shocked

def _build_rf_shock_df(rf_needed: list[str],
                       instr_indexed: pd.DataFrame,
                       shocked_rf_vals: pd.DataFrame
                       ) -> pd.DataFrame:
    """Build a DataFrame with one column per element in rf_needed."""

    cols = {}

    for rf in rf_needed:

        if rf not in instr_indexed.index:
            raise KeyError(f"Risk factor '{rf}' not found in instr_indexed.index")

        mat = instr_indexed.at[rf, "maturity"]

        # rfr_ccy overrides ccy when the discount curve currency differs from the book currency
        # (e.g. a GBP gilt reported in EUR). It falls back to ccy when rfr_ccy is not set.
        rfr_ccy = instr_indexed.at[rf, "rfr_ccy"] if "rfr_ccy" in instr_indexed.columns else None
        ccy = rfr_ccy if pd.notna(rfr_ccy) else instr_indexed.at[rf, "ccy"]

        # Fall back to the raw maturity label if it is not a plain integer
        try:
            mat_str = f"{int(mat):02d}"
        except Exception:
            mat_str = str(mat)
        shocks_col = f"IR_{ccy}_{mat_str}"
        if shocks_col not in shocked_rf_vals.columns:
            raise KeyError(f"Column '{shocks_col}' not found in shocks")

        # Align each Series to the shocks index automatically
        cols[rf] = shocked_rf_vals[shocks_col]

    return pd.DataFrame(cols)

def _calc_FX_price(mkt_rates: NDArray[np.floating],
                   ) -> NDArray[np.floating]:
    """Get the FX rate quoted in EUR as of the reference date."""

    return mkt_rates

def _calc_EQ_price(mkt_rates: NDArray[np.floating]
                   ) -> NDArray[np.floating]:
    """Get the equity market price as of the reference date."""

    return mkt_rates

def _calc_ZCB_price(face_vals: NDArray[np.floating],
                    maturities: NDArray[np.floating],
                    rfr: NDArray[np.floating]
                    ) -> NDArray[np.floating]:
    """Price a zero-coupon bond excl. inflation and credit risk."""

    # Continuous compounding
    disc_fact = np.exp(-rfr * maturities)
    prices = disc_fact * face_vals

    return prices

def _calc_ZCB_INFL_price(face_vals: NDArray[np.floating],
                         maturities: NDArray[np.floating],
                         rfr: NDArray[np.floating],
                         set_infl: NDArray[np.floating]
                         ) -> NDArray[np.floating]:
    """Price an inflation-linked zero-coupon bond excl. credit risk."""

    infl_fact = (1 + set_infl) ** maturities
    infl_adj_face_vals = face_vals * infl_fact

    # Discrete compounding
    disc_fact = 1 / (1 + rfr) ** maturities
    prices = disc_fact * infl_adj_face_vals

    return prices

def _calc_ZCB_CS_price(face_vals: NDArray[np.floating],
                       maturities: NDArray[np.floating],
                       rfr: NDArray[np.floating],
                       cra_bps: NDArray[np.floating],
                       set_cs: NDArray[np.floating]
                       ) -> NDArray[np.floating]:
    """Price a zero-coupon bond with credit risk, excl. inflation."""

    cra = cra_bps / 10E+3

    # Discrete compounding
    disc_fact = 1 / (1 + rfr + cra + set_cs) ** maturities
    prices = disc_fact * face_vals

    return prices

def _calc_CALL_price(spots: NDArray[np.floating],
                     strikes: NDArray[np.floating],
                     maturities: NDArray[np.floating],
                     rfr: NDArray[np.floating],
                     volas: NDArray[np.floating],
                     ) -> NDArray[np.floating]:
    """Calculate Black–Scholes prices for European call options."""

    time_fact = np.sqrt(maturities)

    d1 = (
        np.log(spots / strikes)
        + (rfr + 0.5 * volas**2) * maturities
    ) / (volas * time_fact)
    d2 = d1 - volas * time_fact

    prices = (
        spots * norm.cdf(d1)
        - strikes * np.exp(-rfr * maturities) * norm.cdf(d2)
    )

    return prices

def _calc_PUT_price(spots: NDArray[np.floating],
                    strikes: NDArray[np.floating],
                    maturities: NDArray[np.floating],
                    rfr: NDArray[np.floating],
                    volas: NDArray[np.floating],
                    ) -> NDArray[np.floating]:
    """Calculate Black–Scholes prices for European put options."""

    time_fact = np.sqrt(maturities)

    d1 = (
        np.log(spots / strikes)
        + (rfr + 0.5 * volas**2) * maturities
    ) / (volas * time_fact)
    d2 = d1 - volas * time_fact

    prices = (
        strikes * np.exp(-rfr * maturities) * norm.cdf(-d2)
        - spots * norm.cdf(-d1)
    )

    return prices

def _convert_loc_ccy_to_eur(prices_loc: pd.DataFrame,
                            instr_info: pd.DataFrame
                            ) -> pd.DataFrame:
    """Convert prices quoted in local currency to EUR values."""

    prices_eur = prices_loc.copy()

    for _, row in instr_info.iterrows():

        instr = row["fin_instr"]
        val_tag = row["val_tag"]
        # rfr_ccy overrides ccy when the discount-curve/issuing currency differs from the book
        # currency. It falls back to ccy where rfr_ccy is not set. Uses the same convention as
        # in _build_rf_shock_df / io_helpers._effective_ccy.
        rfr_ccy = row["rfr_ccy"] if "rfr_ccy" in row.index else None
        ccy = rfr_ccy if pd.notna(rfr_ccy) else row["ccy"]

        if val_tag == "FX" or ccy == "EUR":
            continue

        fx_col_candidates = [col for col in prices_loc.columns if col.startswith(f"FX-{ccy}-")]
        if not fx_col_candidates:
            raise KeyError(f"No FX column found for currency '{ccy}' required for '{instr}'")
        fx_col = fx_col_candidates[0]

        prices_eur[instr] = prices_loc[instr] * prices_loc[fx_col]

    return prices_eur

def calc_prices(mkt_data: pd.DataFrame,
                instr_info: pd.DataFrame,
                ref_date: str,
                param_config: dict[str, str],
                der_underlyings: dict[str, str],
                shocks: pd.DataFrame | None = None
                ) -> pd.DataFrame:
    """Run the pricing functions for all financial instruments grouped by val_tag."""

    arg_spec = get_pricing_arg_spec(module=sys.modules[__name__],
                                     prefix="_calc_",
                                     suffix="_price")

    is_base_scenario = shocks is None
    base_rf_vals = _get_mtm_base_value(mkt_data, ref_date)
    if not is_base_scenario:
        shock_types = param_config["valuation"]["shock_type"]
        shocked_rf_vals = _apply_rf_shocks(base_rf_vals, shocks, shock_types)
        final = pd.DataFrame(index=shocks.index)
    else:
        shocked_rf_vals = base_rf_vals
        final = pd.DataFrame(index=base_rf_vals.index)

    val_tags = instr_info["val_tag"].unique()
    for val_tag in val_tags:

        mask = instr_info["val_tag"] == val_tag
        rf_needed = instr_info.loc[mask, "fin_instr"].tolist()

        if not rf_needed:
            continue

        # Pricing functions are identified by the naming convention "_calc_{val_tag}_price"
        price_func = get_pricing_func(tag=val_tag,
                                      module=sys.modules[__name__]
                                      )

        instr_indexed = instr_info.set_index("fin_instr", drop=False)

        def arg_source(arg_name: str, shocked_rf_vals: pd.DataFrame) -> Any:
            """Resolve a pricing function argument by name, from the current scenario state."""

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
                # Strikes are set at inception, so they come from the base (unshocked) spots
                underlying_cols = [der_underlyings[d] for d in rf_needed]
                strikes_data = base_rf_vals[underlying_cols].to_numpy()
                return strikes_data[0]

            elif arg_name == "maturities":
                return instr_indexed.loc[rf_needed, "maturity"].to_numpy(dtype=float)

            elif arg_name == "rfr":
                return _build_rf_shock_df(rf_needed=rf_needed,
                                              instr_indexed=instr_indexed,
                                              shocked_rf_vals=shocked_rf_vals
                                              ).to_numpy()

            elif arg_name == "volas":
                # Implied vola is looked up by the option's own issuer: "EQVOL_{issuer_short}""
                eqvol_cols = [f"EQVOL_{instr_indexed.at[d, 'issuer_short']}" for d in rf_needed]
                missing = [c for c in eqvol_cols if c not in shocked_rf_vals.columns]
                if missing:
                    raise KeyError(f"No EQVOL risk factor found for column(s) {missing}")
                return shocked_rf_vals[eqvol_cols].to_numpy()

            elif arg_name == "face_vals":
                return instr_indexed.loc[rf_needed, "face_value"].to_numpy(dtype=float)

            elif arg_name == "cra_bps":
                return instr_indexed.loc[rf_needed, "cra (bps)"].to_numpy(dtype=float)

            elif arg_name == "set_cs":
                return instr_indexed.loc[rf_needed, "set_cs"].to_numpy(dtype=float)

            elif arg_name == "set_infl":
                return instr_indexed.loc[rf_needed, "set_infl"].to_numpy(dtype=float)

            elif arg_name == "ref_date":
                return ref_date

            else:
                raise ValueError(f"Unknown argument specifier '{arg_name}'")

        arg_values = [arg_source(arg, shocked_rf_vals) for arg in arg_spec[val_tag]]
        prices = price_func(*arg_values)
        final.loc[:, rf_needed] = prices

    final = _convert_loc_ccy_to_eur(final, instr_info)

    return final

def comp_prices_with_calib_targets(base_values: pd.DataFrame,
                                   calib_target: pd.DataFrame
                                   ) -> None:
    """Check whether the base values are close enough to the calibration targets."""

    eps = 10E-5
    exceeds = (base_values["price"] - calib_target["calib_target"]).abs() > eps

    if exceeds.any():
        print("The base prices of the following instruments deviate strongly from the"
              "EIOPA calibration targets:")
        print(base_values.loc[exceeds, "fin_instr"].tolist())
    else:
        print("All base prices are close to the EIOPA calibration targets.")

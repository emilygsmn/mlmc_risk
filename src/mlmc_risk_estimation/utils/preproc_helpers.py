"""Module providing data preprocessing helper functions."""

import pandas as pd

from utils.io_helpers import import_hist_market_data, import_riskfree_rates_from_file

__all__ = ["preproc_portfolio", "get_historical_data"]

def _select_port_instr(port, instr_info):
    """Function selecting the relevant portfolio positions."""

    # Select the relevant (non-zero) positions from portfolio
    port = port[port.iloc[:,1] != 0.0]

    # Filter for relevant instrument data from meta dataframe
    instr_list = port.iloc[:, 0].tolist()
    instr_info = instr_info[instr_info.iloc[:, 0].isin(instr_list)]

    return port, instr_info

def _add_valuation_tag(instr_info):
    """Function categorizing the instruments by valuation method."""
    # BOND, BOND_FX, BOND_IF, BOND_IF_FX, BOND_CS, BOND_CS_FX, BOND_IF_CS, BOND_IF_CS_FX
    # DER_SWAP, DER_PUT
    # FX
    # ...

    # Define the conditions to assign the valuation tags by
    def _classify(name):
        if name.startswith("FX"):
            return "FX"
        if name.startswith("Other-EQ"):
            return "EQ"
        if "FI" in name:
            if name.startswith("GOV-FI-"):
                return "BOND"
            if name == "FI-GBP-RFR-NA-NA-NA-NA-01" or name == "GOV-FI-UK-NA-NA-05":
                return "BOND_FX"
        else:
            return "unknown"

    # Apply the classification to all instruments and save the tag in new column
    instr_info["val_tag"] = instr_info["fin_instr"].apply(_classify)

    return instr_info

def _get_calib_target(instr_info):
    """Function selecting the calibration targets from the instrument meta data."""
    return instr_info[["fin_instr", "calibration_target"]]

def preproc_portfolio(port, instr_info):
    """Function preprocessing the portfolio composition and instrument meta data."""

    # Select only the non-zero components from the portfolio
    #port, instr_info = _select_port_instr(port, instr_info)

    ### Preliminary filtering only for testing purposes:
    selected_positions = [
        "GOV-FI-AT-NA-NA-05",
        "GOV-FI-AT-NA-NA-10",
        "GOV-FI-AT-NA-NA-20",
        #"GOV-FI-UK-NA-NA-05",
        #"FI-GBP-RFR-NA-NA-NA-NA-01",
        "Other-EQ-EUR-PUBL-EU-SX5T-NA-NA-NA",
        "Other-EQ-EUR-PUBL-EU-MSDEE15N-NA-NA-NA",
        #"Other-EQ-EUR-PUBL-UK-TUKXG-NA-NA-NA",
        "Other-EQ-EUR-PUBL-US-SPTR500N-NA-NA-NA",
        "FX-GBP-NA-NA-NA-NA-NA-NA",
        "FX-USD-NA-NA-NA-NA-NA-NA"
    ]
    port = port[port["fin_instr"].isin(selected_positions)]
    instr_info = instr_info[instr_info["fin_instr"].isin(selected_positions)]
    ###

    # Add a valuation tag
    instr_info = _add_valuation_tag(instr_info)

    # Get calibration target
    calib_target = _get_calib_target(instr_info)

    return port, instr_info, calib_target

def merge_ecb_with_yf(df_ecb: pd.DataFrame, df_yf: pd.DataFrame) -> pd.DataFrame:
    """Function merging historical data from ECB and Yahoo! Finance (only for common dates)."""

    # Ensure both DataFrames have DatetimeIndex
    def ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index, errors="raise")
        # remove timezone info to avoid mismatches
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        # sort index
        df = df.sort_index()
        return df

    df_ecb = ensure_dt_index(df_ecb.copy())
    df_yf  = ensure_dt_index(df_yf.copy())

    # Compute their common dates (intersection)
    common_idx = df_ecb.index.intersection(df_yf.index)

    # Handle the case where no common dates exist
    if common_idx.empty:
        raise ValueError("No overlapping dates between ECB and yfinance dataframes.")

    # Select only common dates
    a = df_ecb.loc[common_idx]
    b = df_yf.loc[common_idx]

    # Concatenate columns side-by-side
    merged = pd.concat([a, b], axis=1)

    # Ensure index is unique and sorted
    merged = merged.loc[~merged.index.duplicated(keep='first')]
    merged = merged.sort_index()

    return merged

def get_historical_data(path_config, param_config, instr_info):
    """Function importing all relevant historical data from different sources and merging them."""

    # Get market data from yahoo! finance
    market_data = import_hist_market_data(param_config, instr_info)

    # Get risk-free spot rate yield curves from files
    rfr_data = import_riskfree_rates_from_file(path_config["input"], instr_info)

    # Merge historical data from ECB and Yahoo! Finance
    hist_data = merge_ecb_with_yf(df_yf=market_data,
                                  df_ecb=rfr_data)
    return hist_data

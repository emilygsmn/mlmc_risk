"""Module providing data preprocessing helper functions."""

import pandas as pd

from mlmc_risk_estimation.portfolio.utils.io_helpers import (import_hist_market_data,
                                                    import_eqvol_data,
                                                    import_riskfree_rates_from_file,
                                                    import_boe_gilt_data)

__all__ = ["preproc_portfolio", "get_historical_data"]

def _calc_position_weights(portfolio: pd.DataFrame,
                           instr_info: pd.DataFrame,
                           target_total: float = 1000.0
                           ) -> pd.Series:
    """Compute per-instrument P&L multipliers (position units, scaled up by a
       single constant scalar so the held FUNDED-ASSET instruments' calibration-target market
       value sums to target_total, compensating for benchmark constituents that are not 
       included due to missing data).

       Derivatives and FX (instr_type "DER" or "FX"}) are notional overlay positions with
       approximately zero funded value at inception, thereby not part of the portfolio's NAV.
    """

    bm_col = portfolio.columns[1]
    merged = portfolio.merge(instr_info[["fin_instr", "instr_type", "calibration_target"]],
                             on="fin_instr")
    is_overlay = merged["instr_type"].isin(["DER", "FX"])
    market_value = (merged[bm_col] * merged["calibration_target"]).where(~is_overlay)
    scale_factor = target_total / market_value.sum()

    return merged.set_index("fin_instr")[bm_col] * scale_factor

def _select_port_instr(portfolio: pd.DataFrame,
                       instr_info: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the relevant portfolio positions."""

    # Select the relevant (non-zero) positions from the portfolio
    portfolio = portfolio[portfolio.iloc[:, 1] != 0.0]

    instr_list = portfolio.iloc[:, 0].tolist()
    instr_info = instr_info[instr_info.iloc[:, 0].isin(instr_list)]

    return portfolio, instr_info

def _add_valuation_tag(instr_info: pd.DataFrame) -> pd.DataFrame:
    """Categorize the instruments by valuation method."""

    def _classify(name: str) -> str | None:
        """Map an instrument name to its val_tag by naming-convention keyword."""
        if name.startswith("FX"):
            return "FX"
        if name.startswith("Other-EQ"):
            return "EQ"
        if name.startswith("Other-RE"):
            return "RE"
        if "FI" in name:
            if "-INFL-" in name:
                return "ZCB_INFL"
            # Plain risk-free curve bond vs. one requiring a credit spread
            if "-RFR-" in name:
                return "ZCB"
            return "ZCB_CS"
        if name.startswith("DER"):
            if "-SWA-" in name:
                return "SWAP"
            if "EQ-PUT" in name:
                return "PUT"
            if "EQ-CALL" in name:
                return "CALL"
        else:
            return None

    instr_info["val_tag"] = instr_info["fin_instr"].apply(_classify)

    return instr_info

def _get_calib_target(instr_info: pd.DataFrame) -> pd.DataFrame:
    """Select the calibration targets from the instrument meta data."""

    return instr_info[["fin_instr", "calibration_target"]]

def _map_derivative_underlyings(instr_info: pd.DataFrame) -> dict[str, str]:
    """Map the derivatives to their underlying assets."""

    map_dict: dict[str, str] = {}

    # Equity derivatives, matched to their underlying by issuer abbreviation
    der_mask = (
        instr_info["fin_instr"].str.startswith("DER", na=False)
        & instr_info["sector_level_1"].str.contains("EQ", na=False)
    )
    derivatives = instr_info.loc[der_mask]

    underlying_mask = instr_info["instr_type"].str.contains("Other-EQ", na=False)
    underlyings = instr_info.loc[underlying_mask, ["issuer_short", "fin_instr"]]

    for _, der_row in derivatives.iterrows():
        issuer = der_row["issuer_short"]

        match = underlyings.loc[
            underlyings["issuer_short"] == issuer, "fin_instr"
        ]

        if match.empty:
            raise ValueError(
                f"No underlying instrument found for derivative "
                f"{der_row['fin_instr']} (issuer_short='{issuer}')"
            )

        map_dict[der_row["fin_instr"]] = match.iloc[0]

    return map_dict

def preproc_portfolio(portfolio: pd.DataFrame,
                      instr_info: pd.DataFrame
                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], pd.Series]:
    """Preprocess the portfolio composition and instrument meta data."""

    # Select only the non-zero-weight components from the reduced MCRCS file.
    portfolio, instr_info = _select_port_instr(portfolio, instr_info)

    # Alias the raw source column once, instead of repeating it in every consumer
    instr_info = instr_info.rename(columns={"notional_/_pos_units": "face_value"})

    instr_info = _add_valuation_tag(instr_info)
    der_underlyings = _map_derivative_underlyings(instr_info)
    weights = _calc_position_weights(portfolio, instr_info)

    return portfolio, instr_info, der_underlyings, weights

def merge_hist_data(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Merge historical data from multiple sources (only for common dates)."""

    def ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a DataFrame to a sorted, timezone-naive DatetimeIndex."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index, errors="raise")
        # Drop timezone info, otherwise the index intersection below can miss some matches
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        return df.sort_index()

    dfs = [ensure_dt_index(df.copy()) for df in dfs]

    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)
    if common_idx.empty:
        raise ValueError("No overlapping dates between the historical data sources.")

    merged = pd.concat([df.loc[common_idx] for df in dfs], axis=1)
    merged = merged.loc[~merged.index.duplicated(keep="first")]
    merged = merged.sort_index()

    return merged

def _rebase_eq_prices(hist_data: pd.DataFrame,
                      instr_info: pd.DataFrame,
                      val_date: str
                      ) -> pd.DataFrame:
    """Rescale each EQ price series onto the EIOPA calibration_target level.

    Our EQ tickers are the closest openly available proxies (ETF shares or differently-rebased indices),
    not the exact instruments calibration_target refers to, so their absolute levels are off.
    Rescaling the whole series to match at val_date preserves the proxy's
    own return dynamics while making its market value consistent with the EIOPA weighting.
    """

    eq_instr = instr_info.loc[instr_info["val_tag"] == "EQ", ["fin_instr", "calibration_target"]]
    hist_data = hist_data.copy()
    for _, row in eq_instr.iterrows():
        fin_instr, calibration_target = row["fin_instr"], row["calibration_target"]
        if fin_instr in hist_data.columns:
            factor = calibration_target / hist_data.loc[val_date, fin_instr]
            hist_data[fin_instr] *= factor

    return hist_data

def get_historical_data(path_config: dict,
                        param_config: dict,
                        instr_info: pd.DataFrame
                        ) -> pd.DataFrame:
    """Import all relevant historical data from different sources and merge it into one frame."""

    market_data = import_hist_market_data(param_config, instr_info)
    rfr_data = import_riskfree_rates_from_file(path_config["input"], instr_info)
    eqvol_data = import_eqvol_data(param_config, path_config["input"])
    boe_gilt_data = import_boe_gilt_data(path_config["input"], instr_info)

    sources = [df for df in (market_data, rfr_data, eqvol_data, boe_gilt_data) if df is not None]
    hist_data = merge_hist_data(*sources)

    val_date = param_config["valuation"]["val_date"]
    return _rebase_eq_prices(hist_data, instr_info, val_date)

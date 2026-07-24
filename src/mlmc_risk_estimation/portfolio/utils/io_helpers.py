"""Module providing input/output handling functions."""

import yaml
import pandas as pd
import yfinance as yf

__all__ = ["read_config",
           "get_portfolio",
           "get_instr_info",
           "import_hist_market_data",
           "import_eqvol_data",
           "import_riskfree_rates_from_file",
           "import_boe_gilt_data"
           ]

def read_config(file_path: str) -> dict:
    """Read the configuration parameters from a YAML file."""

    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def _import_mcrcs_data(file_path: str, excel_sheet: str, skip_row: int) -> pd.DataFrame:
    """Import the EIOPA benchmark portfolios."""

    data = pd.read_excel(
        file_path,
        sheet_name=excel_sheet,
        skiprows=skip_row,
        header=0
    )

    data = (
        data
        .loc[:, ~data.columns.str.contains("^Unnamed")]
        .dropna(axis=1, how="all")
        .rename(columns={data.columns[0]: "fin_instr"})
        )

    return data

def get_portfolio(input_config: dict, param_config: dict) -> pd.DataFrame:
    """Import the selected MCRCS benchmark portfolio data."""

    portfolio = _import_mcrcs_data(file_path=input_config["mcrcs_data"],
                                   excel_sheet=input_config["portfolio_data"]["worksheet"],
                                   skip_row=input_config["portfolio_data"]["rows_to_skip"]
                                   )

    return portfolio[["fin_instr", param_config["valuation"]["bm_portfolio"]]]

def get_instr_info(input_config: dict) -> pd.DataFrame:
    """Import the data on the MCRCS financial instruments."""

    instr_info = _import_mcrcs_data(file_path=input_config["mcrcs_data"],
                                    excel_sheet=input_config["instrument_data"]["worksheet"],
                                    skip_row=input_config["instrument_data"]["rows_to_skip"]
                                    )

    return instr_info

def _get_yf_ticker(ticker_map: dict[str, str], instr_info: pd.DataFrame) -> list[str]:
    """Map instrument names to their Yahoo! Finance tickers."""

    return (
        instr_info["fin_instr"]
        .map(ticker_map)
        .dropna()
        .tolist()
    )

def import_hist_market_data(param_config: dict, instr_info: pd.DataFrame) -> pd.DataFrame:
    """Download historical market time series data from Yahoo! Finance."""

    ticker_map = param_config["valuation"]["yf_ticker_map"]
    tickers = _get_yf_ticker(ticker_map=ticker_map,
                             instr_info=instr_info
                             )

    # Forward-fill gaps so no future price leaks backward; only fall back to
    # backward-fill for leading NaNs where no prior value exists yet
    mkt_data = (yf.download(tickers,
                           start=param_config["valuation"]["hist_data_start"],
                           end=param_config["valuation"]["hist_data_end"]
                           )["Close"]
                .ffill()
                .bfill()
                )

    rev_map = {v: k for k, v in ticker_map.items()}
    mkt_data.rename(columns=rev_map, inplace=True)

    return mkt_data

def import_eqvol_data(param_config: dict, input_config: dict) -> pd.DataFrame | None:
    """Download implied-volatility index data for equity volatility risk factors."""

    eqvol_map = input_config.get("eqvol_data")
    if not eqvol_map:
        return None

    tickers = list(eqvol_map.values())
    eqvol_data = (yf.download(tickers,
                             start=param_config["valuation"]["hist_data_start"],
                             end=param_config["valuation"]["hist_data_end"]
                             )["Close"]
                  .ffill()
                  .bfill()
                  )

    # Vola indices are quoted in percentage points, so convert to decimal
    rev_map = {ticker: f"EQVOL_{issuer_short}" for issuer_short, ticker in eqvol_map.items()}
    eqvol_data = eqvol_data.rename(columns=rev_map) / 100

    return eqvol_data

def _effective_ccy(instr_info: pd.DataFrame) -> pd.Series:
    """Per-instrument discount-curve currency: rfr_ccy overrides ccy when the discount curve
    currency differs from the book currency but falls back to ccy where rfr_ccy is not set
    or the column is not there. Same convention as in full_valuation.py's _build_rf_shock_df."""
    if "rfr_ccy" not in instr_info.columns:
        return instr_info["ccy"]
    return instr_info["rfr_ccy"].where(instr_info["rfr_ccy"].notna(), instr_info["ccy"])

def import_riskfree_rates_from_file(input_config: dict, instr_info: pd.DataFrame) -> pd.DataFrame:
    """Import the risk-free rates from csv files."""

    path_dict = input_config["rfr_data"]
    currencies = list(path_dict.keys())
    rfr_df = None

    disc_ccy = _effective_ccy(instr_info)

    for ccy in currencies:
        ccy_input = path_dict[ccy]
        base = ccy_input["path"]
        date_col = ccy_input["date_col"]
        data_col = ccy_input["data_col"]
        cols_to_read = [date_col, data_col]

        maturities = (
        instr_info
        .loc[disc_ccy == ccy]
        .loc[instr_info["instr_type"] == "FI", "maturity"]
        .astype(int)
        .unique()
        .tolist()
        )
        mats = [f"{int(mat):02d}" for mat in maturities]

        for mat in mats:
            file = base + f"_{mat}y.csv"

            tmp_df = pd.read_csv(file, header=0, usecols=cols_to_read)
            tmp_df.columns = ["date", f"IR_{ccy}_{mat}"]
            tmp_df["date"] = pd.to_datetime(tmp_df["date"])
            tmp_df = tmp_df.set_index("date").sort_index()

            # Source rates are quoted in percent, so convert to decimal
            tmp_df[f"IR_{ccy}_{mat}"] = tmp_df[f"IR_{ccy}_{mat}"] / 100

            if rfr_df is None:
                rfr_df = tmp_df
            else:
                col_name = f"IR_{ccy}_{mat}"

                if not rfr_df.index.equals(tmp_df.index):
                    # Align to the existing dates and forward-fill the gaps
                    aligned = (
                        tmp_df
                        .reindex(rfr_df.index)
                        .ffill()
                    )
                    rfr_df[col_name] = aligned[col_name]
                else:
                    rfr_df[col_name] = tmp_df[col_name]

    return rfr_df

def import_boe_gilt_data(input_config: dict, instr_info: pd.DataFrame) -> pd.DataFrame | None:
    """Import UK gilt spot rates from the Bank of England GLC nominal archive files.

    Each archive file contains a wide "spot curve" sheet: one maturity-year header row, then one
    row per date with a spot rate column per maturity. Only the maturities actually needed by
    GBP-denominated instruments are extracted.
    """

    boe_config = input_config.get("boe_gilt_data")
    if not boe_config:
        return None

    disc_ccy = _effective_ccy(instr_info)
    maturities = (
        instr_info
        .loc[disc_ccy == "GBP"]
        .loc[instr_info["instr_type"] == "FI", "maturity"]
        .astype(int)
        .unique()
        .tolist()
    )
    if not maturities:
        return None

    sheet = boe_config["sheet"]
    maturity_row = boe_config["maturity_row"]
    data_start_row = boe_config["data_start_row"]

    frames = []
    for file in boe_config["files"]:
        header = pd.read_excel(file, sheet_name=sheet, header=None,
                               skiprows=maturity_row, nrows=1).iloc[0]
        col_by_maturity = {float(v): i for i, v in enumerate(header) if pd.notna(v) and i > 0}

        cols_needed = {mat: col_by_maturity[float(mat)]
                      for mat in maturities if float(mat) in col_by_maturity}
        if not cols_needed:
            continue

        data = pd.read_excel(file, sheet_name=sheet, header=None, skiprows=data_start_row,
                             usecols=[0] + list(cols_needed.values()))
        data.columns = ["date"] + [f"IR_GBP_{mat:02d}" for mat in cols_needed]
        frames.append(data)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.drop_duplicates(subset="date").sort_values("date").set_index("date")

    # Bank holidays are present as all-NaN rows in the source. Forward-fill (falling back to
    # backward-fill only for leading NaNs)
    combined = combined.ffill().bfill()

    # Rates are quoted in percent, so convert to decimal
    return combined / 100

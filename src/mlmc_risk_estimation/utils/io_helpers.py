"""Module providing input/output handling functions."""

import yaml
import pandas as pd
import yfinance as yf

__all__ = ["read_config",
           "get_portfolio",
           "get_instr_info",
           "import_hist_market_data",
           "import_riskfree_rates_from_file"
           ]

def read_config(file_path):
    """Function reading the configuration parameters from a YAML file."""

    # Read the configurations from the yaml file
    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def _import_mcrcs_data(file_path, excel_sheet, skip_row):
    """Function importing the EIOPA benchmark portfolios."""

    # Read the benchmark portfolio data from the csv file
    portfolio_df =  pd.read_excel(
        file_path,
        sheet_name=excel_sheet,
        skiprows=skip_row,
        header=0
    )

    portfolio_df = (
        portfolio_df
        .loc[:, ~portfolio_df.columns.str.contains("^Unnamed")]
        .dropna(axis=1, how="all")
        .rename(columns={portfolio_df.columns[0]: "fin_instr"})
        )

    return portfolio_df

def get_portfolio(input_config, param_config):
    """Function importing the selected MCRCS benchmark portfolio data."""

    # Import benchmark portfolio data from the csv file
    portfolio_data = _import_mcrcs_data(file_path=input_config["mcrcs_data"],
                                        excel_sheet=input_config["portfolio_data"]["worksheet"],
                                        skip_row=input_config["portfolio_data"]["rows_to_skip"]
                                        )

    return portfolio_data[["fin_instr", param_config["valuation"]["bm_portfolio"]]]

def get_instr_info(input_config):
    """Function importing the data on the MCRCS financial instruments."""

    # Import financial instrument information from the csv file
    instrument_data = _import_mcrcs_data(file_path=input_config["mcrcs_data"],
                                         excel_sheet=input_config["instrument_data"]["worksheet"],
                                         skip_row=input_config["instrument_data"]["rows_to_skip"]
                                         )

    return instrument_data

def _get_yf_ticker(ticker_map, instr_info):
    return (
        instr_info["fin_instr"]
        .map(ticker_map)
        .dropna()
        .tolist()
    )

def import_hist_market_data(param_config, instr_info):
    """Function downloading historical market time series data from Yahoo! Finance."""

    # Get a list of all tickers to retrieve time series data for
    ticker_map = param_config["valuation"]["yf_ticker_map"]
    tickers = _get_yf_ticker(ticker_map=ticker_map,
                             instr_info=instr_info
                             )

    # Download data vie the Yahoo! Finance API
    mkt_data = (yf.download(tickers,
                           start=param_config["valuation"]["hist_data_start"],
                           end=param_config["valuation"]["hist_data_end"]
                           )["Close"]
                .bfill()
                )

    rev_map = {v: k for k, v in ticker_map.items()}

    mkt_data.rename(columns=rev_map, inplace=True)

    return mkt_data

def import_riskfree_rates_from_file(input_config, instr_info):
    """Function importing the ECB risk-free rates from csv files."""

    # Get the base file path
    base = input_config["ecb_rfr_data"]

    # Select the maturities needed for EUR government bonds
    maturities = (
      instr_info
      .loc[instr_info["instr_type"] == "FI", "maturity"]
      .astype(int)
      .unique()
      .tolist()
    )
    mats = [f"{int(m):02d}" for m in maturities]

    # Initialize the data frame to collect the time series in
    rfr_df = None

    # Loop over all relevant maturities
    for m in mats:
        # Create complete file path
        file = base + f"_{m}y.csv"

        # Read the time series from the csv
        tmp_df = pd.read_csv(file, header=0, usecols=[0, 2])
        tmp_df.columns = ["date", f"IR_EUR_{m}"]
        tmp_df["date"] = pd.to_datetime(tmp_df["date"])
        tmp_df = tmp_df.set_index("date").sort_index()

        # Only add the new data to the df if the dates match
        if rfr_df is None:
            rfr_df = tmp_df
        else:
            # Ensure that the dates match
            if not rfr_df.index.equals(tmp_df.index):
                raise ValueError(f"Date mismatch in file {file}")
            rfr_df[f"IR_EUR_{m}"] = tmp_df[f"IR_EUR_{m}"]

    return rfr_df

"""Module providing input/output handling functions."""

import yaml
import pandas as pd
import yfinance as yf

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

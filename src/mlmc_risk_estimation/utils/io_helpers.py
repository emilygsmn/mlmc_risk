"""Module providing input/output handling functions."""

import yaml
import pandas as pd
import yfinance as yf

def _read_config(file_path):
    """Function reading the configuration parameters from a YAML file."""

    # Read the configurations from the yaml file
    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def _import_portfolio_data(file_path, excel_sheet):
    """Function importing the EIOPA benchmark portfolios."""

    # Read the benchmark portfolio data from the csv file
    portfolio_df =  pd.read_excel(
        file_path,
        sheet_name=excel_sheet,
        skiprows=8,
        header=0
    )

    return portfolio_df.rename(columns={portfolio_df.columns[0]: "fin_position"})

def get_portfolio(file_path, excel_sheet, bmp_name):
    """Function importing the selected benchmark portfolio data."""

    # Import benchmakr portfolio data from the csv file
    portfolio_data = _import_portfolio_data(file_path, excel_sheet)

    return portfolio_data[["fin_position", bmp_name]]

def _import_hist_market_data(instr_list, start_date, end_date):
    """Function downloading historical market time series data from Yahoo! Finance."""

    # Get a list of all tickers to retrieve time series data for
    tickers = list(instr_list.keys())

    return yf.download(tickers, start=start_date, end=end_date)["Close"]

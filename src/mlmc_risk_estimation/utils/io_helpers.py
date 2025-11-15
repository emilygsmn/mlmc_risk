"""Module providing input/output handling functions."""

import yaml
import pandas as pd
import yfinance as yf

def _read_config(file_path):
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

    return portfolio_df.rename(columns={portfolio_df.columns[0]: "fin_position"})

def get_portfolio(file_path, port_xl_sheet, port_skpr, bmp_name):
    """Function importing the selected MCRCS benchmark portfolio data."""

    # Import benchmark portfolio data from the csv file
    portfolio_data = _import_mcrcs_data(file_path, port_xl_sheet, port_skpr)

    return portfolio_data[["fin_position", bmp_name]]

def get_instr_info(file_path, instr_xl_sheet, instr_skpr):
    """Function importing the data on the MCRCS financial instruments."""

    # Import financial instrument information from the csv file
    instrument_data = _import_mcrcs_data(file_path, instr_xl_sheet, instr_skpr)

    return instrument_data

def _import_hist_market_data(instr_list, start_date, end_date):
    """Function downloading historical market time series data from Yahoo! Finance."""

    # Get a list of all tickers to retrieve time series data for
    tickers = list(instr_list.keys())

    return yf.download(tickers, start=start_date, end=end_date)["Close"]

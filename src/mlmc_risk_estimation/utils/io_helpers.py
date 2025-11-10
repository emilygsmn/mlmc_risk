"""Module providing input/output handling functions."""

import yaml
import pandas as pd

def _read_config(file_path):
    """Function reading the configuration parameters from a YAML file."""
    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def _import_portfolio_data(file_path, excel_sheet):
    """Function importing the EIOPA benchmark portfolios."""
    portfolio_df =  pd.read_excel(
        file_path,
        sheet_name=excel_sheet,
        skiprows=8,
        header=0
    )
    return portfolio_df.rename(columns={portfolio_df.columns[0]: "fin_position"})

def get_portfolio(file_path, excel_sheet, bmp_name):
    """Function importing the selected benchmark portfolio data."""
    portfolio_data = _import_portfolio_data(file_path, excel_sheet)
    return portfolio_data[["fin_position", bmp_name]]

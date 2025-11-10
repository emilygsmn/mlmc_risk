"""Script for Multilevel Monte Carlo estimation of the Value-at-Risk of a financial portfolio."""

from pathlib import Path

from utils.io_helpers import _read_config, get_portfolio

# Set project root environment variable
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Get input and output paths from path.yaml config file
path_config_dir = PROJECT_ROOT / "data/config/path.yaml"
path_config = _read_config(path_config_dir)

# Get parameter configs from yaml file
param_config_dir = path_config["input"]["param_config"]
param_config = _read_config(param_config_dir)

port_data_dir = path_config["input"]["portfolio_data"]
port_data_sheet = path_config["input"]["portfolio_data_worksheet"]
bm_port_name = param_config["valuation"]["bm_portfolio"]
portfolio = get_portfolio(port_data_dir, port_data_sheet, bm_port_name)
"""Script for Multilevel Monte Carlo estimation of the Value-at-Risk of a financial portfolio."""

from pathlib import Path

from utils.io_helpers import (
    _read_config,
    get_portfolio,
    get_instr_info,
    _import_hist_market_data
)
from scenario_generation import generate_mc_shocks_pycopula


### 1. Read the inputs ###

# Set project root environment variable
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Get input and output paths from path.yaml config file
path_config_dir = PROJECT_ROOT / "data/config/path.yaml"
path_config = _read_config(path_config_dir)

# Get parameter configs from yaml file
param_config_dir = path_config["input"]["param_config"]
param_config = _read_config(param_config_dir)

# Get benchmark portfolio data from csv file
port_data_dir = path_config["input"]["mcrcs_data"]
port_data_sheet = path_config["input"]["portfolio_data"]["worksheet"]
port_skpr = path_config["input"]["portfolio_data"]["rows_to_skip"]
bm_port_name = param_config["valuation"]["bm_portfolio"]
portfolio = get_portfolio(port_data_dir, port_data_sheet, port_skpr, bm_port_name)

instr_data_sheet = path_config["input"]["instrument_data"]["worksheet"]
instr_skpr = path_config["input"]["instrument_data"]["rows_to_skip"]
instr_info = get_instr_info(port_data_dir, instr_data_sheet, instr_skpr)

# Get market data from yahoo finance
market_data = _import_hist_market_data(param_config["valuation"]["tickers"])


### 2. Calibrate the model ###

# Estimate the calibration parameters from the historical data
calib_param = params = {
    "IR": dict(mu=0.02, 
               sigma=0.01,
               dt=1,
               shift=0.025),
    "FX": dict(mu=0.00, 
               sigma=0.07,
               dt=1),
    "EQ": dict(mu=0.05, 
               sigma=0.20,
               dt=1)
}

### 3. Generate Monte Carlo scenarios ###

# Generate Monte Carlo real-world scenario shocks
num_scen = param_config["monte_carlo"]["n"]
stoch_proc_map = param_config["valuation"]["stoch_proc_map"]
mc_scenarios = generate_mc_shocks_pycopula(market_data, stoch_proc_map, calib_param, num_scen)


### 4. Aggregate the profit-and-loss ###



### 5. Estimate the Value-at-Risk ###

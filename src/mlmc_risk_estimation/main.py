"""Script for Multilevel Monte Carlo estimation of the Value-at-Risk of a financial portfolio."""

from pathlib import Path

from utils.io_helpers import (
    read_config,
    get_portfolio,
    get_instr_info,
    import_hist_market_data
)
from utils.preproc_helpers import preproc_portfolio
from scenario_generation import generate_mc_shocks_pycopula
from full_valuation import calc_prices, comp_prices_with_calib_targets

####################################################################################################
### 1. Read the inputs ###
####################################################################################################

# Set project root environment variable
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Get input and output paths from path.yaml config file
path_config_dir = PROJECT_ROOT / "data/config/path.yaml"
path_config = read_config(path_config_dir)

# Get parameter configs from yaml file
param_config_dir = path_config["input"]["param_config"]
param_config = read_config(param_config_dir)

# Get benchmark portfolio data from csv file
portfolio = get_portfolio(path_config["input"], param_config)
instr_info = get_instr_info(path_config["input"])

# Preprocess the benchmark portfolio data
portfolio, instr_info, calib_target = preproc_portfolio(portfolio,
                                                        instr_info)

# Get market data from yahoo! finance
market_data = import_hist_market_data(param_config, instr_info)

####################################################################################################
### 2. Calibrate the model ###
####################################################################################################

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

####################################################################################################
### 3. Price the portfolio instruments ###
####################################################################################################

# Compute the prices of the instruments at the reference date (base values)

# Check if the imported/computed base values are close to the calibration targets
comp_prices_with_calib_targets(base_values, calib_target)

####################################################################################################
### 4. Generate Monte Carlo scenarios ###
####################################################################################################

# Generate Monte Carlo real-world scenario shocks
mc_scenarios = generate_mc_shocks_pycopula(param_config, market_data, calib_param)

####################################################################################################
### 5. Compute scenario losses ###
####################################################################################################

####################################################################################################
### 6. Aggregate the profit-and-loss ###
####################################################################################################

####################################################################################################
### 7. Estimate the Value-at-Risk ###
####################################################################################################

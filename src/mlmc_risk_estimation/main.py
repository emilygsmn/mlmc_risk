"""Script for Multilevel Monte Carlo estimation of the Value-at-Risk of a financial portfolio."""

from pathlib import Path
import pandas as pd

from utils.io_helpers import (
    read_config,
    get_portfolio,
    get_instr_info,
    import_hist_market_data,
    import_riskfree_rates_from_file
)
from utils.preproc_helpers import preproc_portfolio
from stochproc_calibration import calibrate_models
from scenario_generation import generate_mc_shocks_pycopula
from full_valuation import calc_prices, comp_prices_with_calib_targets

def main():

    ################################################################################################
    ### 1. Read the inputs ###
    ################################################################################################

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

    # Get risk-free spot rate yield curves from files
    rfr_data = import_riskfree_rates_from_file(path_config["input"], instr_info)


    ################################################################################################
    ### 2. Calibrate the model ###
    ################################################################################################

    calib_param = {
        "FX": dict(mu=0.00, sigma=0.07, dt=1),
        "Other-EQ-EUR-PUBL-EU-SX5T-NA-NA-NA": dict(mu=0.02, sigma=0.01, dt=1, shift=0.025),
        "Other-EQ-EUR-PUBL-US-SPTR500N-NA-NA-NA": dict(mu=0.05, sigma=0.02, dt=1, shift=0.025),
        "EQ": dict(mu=0.05, sigma=0.20, dt=1)
    }
    # Convert to DataFrame
    calib_param = pd.DataFrame.from_dict(calib_param, orient="columns")

    ################################################################################################
    ### 3. Price the portfolio instruments ###
    ################################################################################################

    # Compute the prices of the instruments at the reference date (base values)
    val_date = param_config["valuation"]["val_date"]
    base_values = calc_prices(mkt_data=market_data,
                            instr_info=instr_info,
                            ref_date=val_date,
                            shocks=None
                            )
    print("Base values:")
    print(base_values)

    # Check if the imported/computed base values are close to the calibration targets
    #comp_prices_with_calib_targets(base_values, calib_target)

    ################################################################################################
    ### 4. Generate Monte Carlo scenarios ###
    ################################################################################################

    calib_param = calibrate_models(market_data, instr_info, param_config)

    # Estimate the calibration parameters from the historical data
    calib_param = {
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

    # Generate Monte Carlo real-world scenario shocks
    mc_scenarios = generate_mc_shocks_pycopula(market_data, instr_info, param_config, calib_param)

    shocked_values = calc_prices(mkt_data=market_data,
                            instr_info=instr_info,
                            ref_date=val_date,
                            shocks=mc_scenarios
                            )
    print("Shocked values:")
    print(shocked_values)

    ################################################################################################
    ### 5. Compute scenario losses ###
    ################################################################################################

    ################################################################################################
    ### 6. Aggregate the profit-and-loss ###
    ################################################################################################

    ################################################################################################
    ### 7. Estimate the Value-at-Risk ###
    ################################################################################################

if __name__ == "__main__":
    main()

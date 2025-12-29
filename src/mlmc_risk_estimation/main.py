"""Script for Multilevel Monte Carlo estimation of the Value-at-Risk of a financial portfolio."""

from pathlib import Path
import pandas as pd

from utils.io_helpers import read_config, get_portfolio, get_instr_info
from utils.preproc_helpers import preproc_portfolio, get_historical_data
from model_calibration import calibrate_models
from scenario_generation import generate_mc_shocks_pycopula
from full_valuation import calc_prices, comp_prices_with_calib_targets
from deltagamma_valuation import _get_greeks, calc_delta_scenario_pnl
from risk_aggregation import (
    calc_instr_pnls,
    calc_portfolio_pnl,
    calc_standard_mc_hd_var
)

def main():
    """Function estimating the Value-at-Risk of a given financial portfolio
       using a copula and Monte Carlo simulation.
    """

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
    portfolio, instr_info, calib_target = preproc_portfolio(port=portfolio,
                                                            instr_info=instr_info)

    # Get historical data
    hist_data = get_historical_data(path_config, param_config, instr_info)
    print("historical data:")
    print(hist_data)

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

    instr_info, calib_param = calibrate_models(hist_data, instr_info, param_config)

    # Compute the prices of the instruments at the reference date (base values)
    val_date = param_config["valuation"]["val_date"]
    base_values = calc_prices(mkt_data=hist_data,
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

    #calib_param = calibrate_models(hist_data, instr_info, param_config)

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
    mc_scenarios = generate_mc_shocks_pycopula(hist_data, instr_info, param_config, calib_param)

    shocked_values = calc_prices(mkt_data=hist_data,
                            instr_info=instr_info,
                            ref_date=val_date,
                            shocks=mc_scenarios
                            )
    print("Shocked values:")
    print(shocked_values)

    ################################################################################################
    ### 5. Compute scenario profits-and-losses ###
    ################################################################################################

    instr_scenario_pnls = calc_instr_pnls(prices_at_t1=base_values,
                                          prices_at_t2=shocked_values)
    print("Instrument scenario profit-and-losses:")
    print(instr_scenario_pnls)

    ################################################################################################
    ### 6. Aggregate the profit-and-loss ###
    ################################################################################################

    total_scenario_pnl = calc_portfolio_pnl(instr_pnls=instr_scenario_pnls)
    print("Total scenario profit-and-losses:")
    print(total_scenario_pnl)

    ################################################################################################
    ### 7. Estimate the Value-at-Risk ###
    ################################################################################################

    hd_var = calc_standard_mc_hd_var(vals_df=total_scenario_pnl,
                                     conf_lvl=0.995)
    print("Standard Monte Carlo Harrell-Davis Value-at-Risk:")
    print(hd_var)

    ################################################################################################
    ### 8. Estimate the Delta Value-at-Risk ###
    ################################################################################################

    delta_scenario_pnl = calc_delta_scenario_pnl(mkt_data=hist_data,
                                        instr_info=instr_info,
                                        ref_date=val_date,
                                        scenario_shocks=mc_scenarios)
    
    print("Delta scenario profit-and-losses")
    print(delta_scenario_pnl)

    delta_hd_var = calc_standard_mc_hd_var(vals_df=delta_scenario_pnl,
                                           conf_lvl=0.995)
    
    print("Delta Standard Monte Carlo Harrell-Davis Value-at-Risk:")
    print(delta_hd_var)

if __name__ == "__main__":
    main()

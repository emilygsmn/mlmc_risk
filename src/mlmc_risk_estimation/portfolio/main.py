"""Script for Multilevel Monte Carlo estimation of the Value-at-Risk of a financial portfolio."""

from pathlib import Path

from mlmc_risk_estimation.portfolio.utils.io_helpers import read_config, get_portfolio, get_instr_info
from mlmc_risk_estimation.portfolio.utils.preproc_helpers import preproc_portfolio, get_historical_data
from mlmc_risk_estimation.portfolio.model_calibration import calibrate_models
from mlmc_risk_estimation.portfolio.scenario_generation import generate_mc_shocks_pycopula
from mlmc_risk_estimation.portfolio.full_valuation import calc_prices
from mlmc_risk_estimation.portfolio.deltagamma_valuation import (
    calc_delta_scenario_pnl,
    calc_delta_gamma_scenario_pnl
)
from mlmc_risk_estimation.portfolio.risk_aggregation import (
    calc_instr_pnls,
    calc_portfolio_pnl,
    calc_standard_mc_hd_var,
    calc_tail_dependence_coeff
)

def main():
    """Function estimating the Value-at-Risk of a given financial portfolio
       using a copula and Monte Carlo simulation.
    """

    # 1. Read the inputs

    # Set project root environment variable
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    portfolio, instr_info, der_underlyings, weights = preproc_portfolio(portfolio=portfolio,
                                                                        instr_info=instr_info)

    # Get historical data
    hist_data = get_historical_data(path_config, param_config, instr_info)
    print("Historical data:")
    print(hist_data)

    # 2. Calibrate the model

    instr_info, calib_param = calibrate_models(hist_data, instr_info, param_config)
    print("Calibration parameters")
    print(calib_param)

    # 3. Price the portfolio instruments

    # Compute the prices of the instruments at the reference date (base values)
    val_date = param_config["valuation"]["val_date"]
    base_values = calc_prices(mkt_data=hist_data,
                              instr_info=instr_info,
                              ref_date=val_date,
                              param_config=param_config,
                              der_underlyings=der_underlyings,
                              shocks=None
                              )
    print("Base values:")
    print(base_values)

    # 4. Generate Monte Carlo scenarios

    mc_scenarios = generate_mc_shocks_pycopula(hist_data, instr_info, param_config, calib_param,
                                               ref_date=val_date)

    shocked_values = calc_prices(mkt_data=hist_data,
                            instr_info=instr_info,
                            ref_date=val_date,
                            param_config=param_config,
                            der_underlyings=der_underlyings,
                            shocks=mc_scenarios,
                            )
    print("Shocked values:")
    print(shocked_values)

    # 5. Compute scenario profits-and-losses

    instr_scenario_pnls = calc_instr_pnls(prices_at_t1=base_values,
                                          prices_at_t2=shocked_values)
    print("Instrument scenario profit-and-losses:")
    print(instr_scenario_pnls)

    # 6. Aggregate the profit-and-loss and switch to the L=-(P&L) convention

    total_scenario_pnl = calc_portfolio_pnl(instr_pnls=instr_scenario_pnls, weights=weights)

    total_scenario_loss = (-total_scenario_pnl).rename(columns={"total_pnl": "total_loss"})
    print("Total scenario losses:")
    print(total_scenario_loss)

    # 7. Estimate the Value-at-Risk

    hd_var = calc_standard_mc_hd_var(vals_df=total_scenario_loss,
                                     conf_lvl=0.995)
    print("Standard Monte Carlo Harrell-Davis Value-at-Risk:")
    print(hd_var)

    # 8. Estimate the delta Value-at-Risk

    delta_scenario_pnl = calc_delta_scenario_pnl(mkt_data=hist_data,
                                                 instr_info=instr_info,
                                                 ref_date=val_date,
                                                 param_config=param_config,
                                                 der_underlyings=der_underlyings,
                                                 scenario_shocks=mc_scenarios,
                                                 weights=weights)
    delta_scenario_loss = (-delta_scenario_pnl).rename(columns={"pnl": "loss"})

    print("Delta scenario losses")
    print(delta_scenario_loss)

    delta_hd_var = calc_standard_mc_hd_var(vals_df=delta_scenario_loss,
                                           conf_lvl=0.995)

    print("Delta Standard Monte Carlo Harrell-Davis Value-at-Risk:")
    print(delta_hd_var)

    # 9. Estimate the delta-gamma Value-at-Risk

    delta_gamma_scenario_pnl = calc_delta_gamma_scenario_pnl(mkt_data=hist_data,
                                        instr_info=instr_info,
                                        ref_date=val_date,
                                        param_config=param_config,
                                        der_underlyings=der_underlyings,
                                        scenario_shocks=mc_scenarios,
                                        weights=weights)
    delta_gamma_scenario_loss = (-delta_gamma_scenario_pnl).rename(columns={"pnl": "loss"})

    print("Delta-Gamma scenario losses")
    print(delta_gamma_scenario_loss)

    delta_gamma_hd_var = calc_standard_mc_hd_var(vals_df=delta_gamma_scenario_loss,
                                           conf_lvl=0.995)

    print("Delta-Gamma Standard Monte Carlo Harrell-Davis Value-at-Risk:")
    print(delta_gamma_hd_var)

    # 10. Compare full-valuation, delta, and delta-gamma losses

    full_loss = total_scenario_loss["total_loss"]
    delta_loss = delta_scenario_loss["loss"]
    delta_gamma_loss = delta_gamma_scenario_loss["loss"]

    full_vs_delta = full_loss.corr(delta_loss)
    full_vs_delta_gamma = full_loss.corr(delta_gamma_loss)

    print("Correlation between full valuation and Delta loss:")
    print(full_vs_delta)
    print("Correlation between full valuation and Delta-Gamma loss:")
    print(full_vs_delta_gamma)

    for conf_lvl in (0.005, 0.01, 0.05):
        print(calc_tail_dependence_coeff(x=full_loss, y=delta_loss, conf_lvl=conf_lvl))
    for conf_lvl in (0.005, 0.01, 0.05):
        print(calc_tail_dependence_coeff(x=full_loss, y=delta_gamma_loss, conf_lvl=conf_lvl))

if __name__ == "__main__":
    main()

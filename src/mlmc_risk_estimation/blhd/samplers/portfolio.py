"""Coupled fine/coarse sampler based on the portfolio valuation framework.

Samples are returned in the loss convention L = -(P&L) (positive = loss, negative = gain), so the
estimator's alpha-quantile is already the VaR.

The fine model is the full-revaluation loss and the coarse model is the delta-gamma approximation
loss. The Greeks do not depend on the scenarios, so they are computed once at setup and reused
across every draw.

Level-1 prices fine and coarse on the same scenario randomness. This ensures the coupling the estimator
relies on.

This is the interface between the `mlmc_risk_estimation.portfolio` package and the BL-HD estimator.
"""

from collections.abc import Callable

import numpy as np

from mlmc_risk_estimation.portfolio.utils.io_helpers import (read_config, get_portfolio,
                                                             get_instr_info)
from mlmc_risk_estimation.portfolio.utils.preproc_helpers import (preproc_portfolio,
                                                                  get_historical_data)
from mlmc_risk_estimation.portfolio.model_calibration import calibrate_models
from mlmc_risk_estimation.portfolio.scenario_generation import generate_mc_shocks_pycopula
from mlmc_risk_estimation.portfolio.full_valuation import calc_prices
from mlmc_risk_estimation.portfolio.deltagamma_valuation import (compute_greeks, apply_delta_pnl,
                                                                 apply_delta_gamma_pnl)
from mlmc_risk_estimation.portfolio.risk_aggregation import calc_instr_pnls, calc_portfolio_pnl

from mlmc_risk_estimation.blhd.samplers.base import make_sampler

__all__ = ["portfolio_sampler"]

def portfolio_sampler(path_config_path: str = "data/config/path.yaml",
                      coarse_model: str = "delta_gamma") -> dict[str, Callable]:
    """Build the portfolio-based coupled sampler dict.

    Loads and calibrates the benchmark portfolio once, precomputes the base values and Greeks,
    and returns {fine, coarse, level0, level1} closures over that state.
    Two coarse models are available:

        "delta_gamma" (default): uses the full delta + gamma quadratic approximation as the
                                 coarse loss
        "delta":                 uses the pure first-order delta approximation
        
    The fine model (full-revaluation) is identical in both, so the pseudo-true VaR and the estimator
    target are unchanged. Only the coarse model, and thereby the fine/coarse coupling, differs.
    """

    if coarse_model not in ("delta_gamma", "delta"):
        raise ValueError(f"coarse_model must be 'delta_gamma' or 'delta', got {coarse_model!r}")
    path_config = read_config(path_config_path)
    param_config = read_config(path_config["input"]["param_config"])

    portfolio, instr_info = get_portfolio(path_config["input"], param_config), \
        get_instr_info(path_config["input"])
    portfolio, instr_info, der_underlyings, weights = preproc_portfolio(portfolio=portfolio,
                                                                        instr_info=instr_info)
    hist_data = get_historical_data(path_config, param_config, instr_info)
    instr_info, calib_param = calibrate_models(hist_data, instr_info, param_config)
    val_date = param_config["valuation"]["val_date"]

    base_values = calc_prices(mkt_data=hist_data, instr_info=instr_info, ref_date=val_date,
                              param_config=param_config, der_underlyings=der_underlyings,
                              shocks=None)
    deltas, gammas = compute_greeks(hist_data, instr_info, val_date, param_config,
                                    der_underlyings, weights)

    def _scenarios(n: int) -> object:
        """Draw n Monte Carlo shock scenarios from the calibrated copula model."""
        param_config["monte_carlo"]["n"] = n
        return generate_mc_shocks_pycopula(hist_data, instr_info, param_config, calib_param,
                                           ref_date=val_date)

    def _fine_loss(scenarios: object) -> np.ndarray:
        """Full-revaluation loss on the scenarios (convention: loss L = -(P&L))."""

        # Loss L = -(P&L): negate the full-revaluation P&L so positive means loss.
        shocked = calc_prices(mkt_data=hist_data, instr_info=instr_info, ref_date=val_date,
                              param_config=param_config, der_underlyings=der_underlyings,
                              shocks=scenarios)
        instr_pnls = calc_instr_pnls(prices_at_t1=base_values, prices_at_t2=shocked)
        return -calc_portfolio_pnl(instr_pnls=instr_pnls, weights=weights).to_numpy().ravel()

    def _coarse_loss(scenarios: object) -> np.ndarray:
        """Coarse approximation loss on the scenarios (delta or delta-gamma)."""

        # Coarse approximation loss (negated P&L): delta-only or delta-gamma per coarse_model.
        if coarse_model == "delta":
            return -apply_delta_pnl(deltas, scenarios).to_numpy().ravel()
        return -apply_delta_gamma_pnl(deltas, gammas, scenarios).to_numpy().ravel()

    def level1(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw n coupled (fine, coarse) losses on the same scenario batch."""
        scenarios = _scenarios(n)
        fine_loss = _fine_loss(scenarios)
        coarse_loss = _coarse_loss(scenarios)
        return fine_loss, coarse_loss, coarse_loss - fine_loss

    def level0(n: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw n coarse-only losses for MLMC level 0."""
        coarse_loss = _coarse_loss(_scenarios(n))
        return coarse_loss, np.zeros_like(coarse_loss)

    def coarse(n: int) -> np.ndarray:
        """Draw n iid coarse (DG) losses."""
        return level0(n)[0]

    def fine(n: int) -> np.ndarray:
        """Draw n iid fine (full-revaluation) losses."""
        return _fine_loss(_scenarios(n))

    return make_sampler(fine=fine, coarse=coarse, level0=level0, level1=level1)

"""Bilevel Harrell-Davis (BL-HD) quantile estimator.

An adaptive multilevel Monte Carlo estimator that combines a cheap coarse model
(the delta or delta-gamma portfolio revaluation, or an analytical surrogate) with
a few expensive fine-model samples to estimate a tail quantile (Value-at-Risk) at
lower cost than fine-only estimators.

This subpackage depends on the subpackage
`mlmc_risk_estimation.portfolio` only through the portfolio sampler.
"""

__all__: list[str] = []

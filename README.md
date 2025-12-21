# Multilevel Monte Carlo for Value-at-Risk Estimation

This project is for testing the efficiency of using Multilevel Monte Carlo methods for estimating the Value-at-Risk of a financial portfolio.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/emilygsmn/mlmc_risk_estimation.git
cd mlmc_risk_estimation
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

## Portfolio, Data & Inputs

The financial portfolios proposed by EIOPA for the Market and Credit Risk Comparative Study YE2024 are used. Instructions and portfolio composition as well as instrument information was taken from https://www.eiopa.europa.eu/browse/supervisory-convergence/internal-models/market-and-credit-risk-comparative-study-ye2024_en?prefLang=es.
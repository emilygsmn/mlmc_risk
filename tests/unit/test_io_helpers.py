"""Module providing functions to test io_helpers.py"""

from unittest.mock import patch

import pandas as pd
from numpy.testing import assert_allclose

from mlmc_risk_estimation.portfolio.utils.io_helpers import (
    read_config,
    _import_mcrcs_data,
    get_portfolio,
    get_instr_info,
    import_riskfree_rates_from_file,
    import_eqvol_data,
    import_boe_gilt_data,
)

def test_read_config_reads_yaml(tmp_path):
    """read_config should parse a YAML file into a dict."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "valuation:\n  hist_data_start: '2020-01-01'\nmonte_carlo:\n  n: 20000\n"
    )

    config = read_config(config_file)

    assert config["valuation"]["hist_data_start"] == "2020-01-01"
    assert config["monte_carlo"]["n"] == 20000

@patch("mlmc_risk_estimation.portfolio.utils.io_helpers.pd.read_excel")
def test_import_mcrcs_data_cleans_columns(mock_read_excel):
    """_import_mcrcs_data should drop Unnamed/all-NaN columns and rename the first column."""
    mock_read_excel.return_value = pd.DataFrame({
        "Code of financial position": ["GOV-FI-AT-05", "GOV-FI-AT-10"],
        "EUR_BMP_01": [8.5, 7.9],
        "Unnamed: 2": [None, None],
        "all_nan_col": [None, None],
    })

    result = _import_mcrcs_data("fake_path.xlsx", "BMP_2024", skip_row=8)

    assert list(result.columns) == ["fin_instr", "EUR_BMP_01"]
    mock_read_excel.assert_called_once_with(
        "fake_path.xlsx", sheet_name="BMP_2024", skiprows=8, header=0
    )

@patch("mlmc_risk_estimation.portfolio.utils.io_helpers._import_mcrcs_data")
def test_get_portfolio_selects_correct_columns(mock_import):
    """get_portfolio should return only fin_instr and the selected benchmark portfolio column."""
    mock_import.return_value = pd.DataFrame({
        "fin_instr": ["GOV-FI-AT-05", "GOV-FI-AT-10"],
        "EUR_BMP_01": [8.5, 7.9],
        "EUR_BMP_02": [9.1, 11.7],
    })
    input_config = {
        "mcrcs_data": "fake_path.xlsx",
        "portfolio_data": {"worksheet": "BMP_2024", "rows_to_skip": 8},
    }
    param_config = {"valuation": {"bm_portfolio": "EUR_BMP_01"}}

    result = get_portfolio(input_config, param_config)

    assert list(result.columns) == ["fin_instr", "EUR_BMP_01"]

@patch("mlmc_risk_estimation.portfolio.utils.io_helpers._import_mcrcs_data")
def test_get_instr_info_returns_full_data(mock_import):
    """get_instr_info should return the imported instrument data unchanged."""
    expected = pd.DataFrame({"fin_instr": ["GOV-FI-AT-05"], "ccy": ["EUR"]})
    mock_import.return_value = expected
    input_config = {
        "mcrcs_data": "fake_path.xlsx",
        "instrument_data": {"worksheet": "Instr_2024_A", "rows_to_skip": 6},
    }

    result = get_instr_info(input_config)

    pd.testing.assert_frame_equal(result, expected)

def test_import_riskfree_rates_from_file_builds_ir_columns(tmp_path):
    """import_riskfree_rates_from_file should build IR_{ccy}_{maturity} columns in decimal."""
    csv_path = tmp_path / "ecb_rfr_05y.csv"
    csv_path.write_text("date,ignored,rate\n2020-01-01,x,1.23\n2020-01-02,x,1.24\n")

    input_config = {
        "rfr_data": {
            "EUR": {"path": str(tmp_path / "ecb_rfr"), "date_col": 0, "data_col": 2},
        }
    }
    instr_info = pd.DataFrame({
        "fin_instr": ["GOV-FI-AT-05"],
        "ccy": ["EUR"],
        "instr_type": ["FI"],
        "maturity": [5],
    })

    result = import_riskfree_rates_from_file(input_config, instr_info)

    # Source percentages 1.23/1.24 are converted to decimal
    assert list(result.columns) == ["IR_EUR_05"]
    assert result["IR_EUR_05"].tolist() == [0.0123, 0.0124]

def test_import_eqvol_data_returns_none_when_not_configured():
    """import_eqvol_data should return None if no eqvol_data is configured."""
    param_config = {"valuation": {"hist_data_start": "2020-01-01", "hist_data_end": "2020-01-03"}}

    assert import_eqvol_data(param_config, {}) is None

@patch("mlmc_risk_estimation.portfolio.utils.io_helpers.yf.download")
def test_import_eqvol_data_renames_and_scales_to_decimal(mock_download):
    """import_eqvol_data should rename tickers to EQVOL_{issuer} and convert percent to decimal."""
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    mock_download.return_value = pd.DataFrame({("Close", "^VIX"): [18.5, 19.0, 20.0]}, index=dates)

    param_config = {"valuation": {"hist_data_start": "2020-01-01", "hist_data_end": "2020-01-03"}}
    input_config = {"eqvol_data": {"SPTR500N": "^VIX"}}

    result = import_eqvol_data(param_config, input_config)

    assert list(result.columns) == ["EQVOL_SPTR500N"]
    assert result["EQVOL_SPTR500N"].tolist() == [0.185, 0.19, 0.20]

def _write_fake_boe_file(path):
    """Helper building a tiny xlsx matching the BoE GLC 'spot curve' sheet layout."""
    rows = [
        ["UK nominal spot curve", None, None],
        [None, None, None],
        ["Maturity", None, None],
        ["years:", 1, 5],
        [None, None, None],
        [None, None, None],
        [pd.Timestamp("2020-01-02"), 1.0, 2.0],
        [pd.Timestamp("2020-01-03"), None, None],   # bank holiday: all-NaN row
        [pd.Timestamp("2020-01-06"), 1.2, 2.2],
    ]
    pd.DataFrame(rows).to_excel(path, sheet_name="4. spot curve", header=False, index=False)

def test_import_boe_gilt_data_returns_none_when_not_configured():
    """import_boe_gilt_data should return None if no boe_gilt_data is configured."""
    instr_info = pd.DataFrame({"fin_instr": [], "ccy": [], "instr_type": [], "maturity": []})

    assert import_boe_gilt_data({}, instr_info) is None

def test_import_boe_gilt_data_returns_none_when_no_gbp_instruments():
    """import_boe_gilt_data should return None if no GBP FI instruments need it."""
    input_config = {
        "boe_gilt_data": {"files": [], "sheet": "x", "maturity_row": 3, "data_start_row": 6}
    }
    instr_info = pd.DataFrame({
        "fin_instr": ["GOV-FI-AT-05"], "ccy": ["EUR"], "instr_type": ["FI"], "maturity": [5],
    })

    assert import_boe_gilt_data(input_config, instr_info) is None

def test_import_boe_gilt_data_extracts_maturities_and_fills_holidays(tmp_path):
    """import_boe_gilt_data should pick the right maturity columns, scale to decimal, and
    forward-fill all-NaN bank-holiday rows."""
    boe_path = tmp_path / "boe.xlsx"
    _write_fake_boe_file(boe_path)

    input_config = {"boe_gilt_data": {
        "files": [str(boe_path)], "sheet": "4. spot curve", "maturity_row": 3, "data_start_row": 6,
    }}
    instr_info = pd.DataFrame({
        "fin_instr": ["FI-GBP-RFR-NA-NA-NA-NA-01", "FI-GBP-RFR-NA-NA-NA-NA-05"],
        "ccy": ["GBP", "GBP"],
        "instr_type": ["FI", "FI"],
        "maturity": [1, 5],
    })

    result = import_boe_gilt_data(input_config, instr_info)

    assert list(result.columns) == ["IR_GBP_01", "IR_GBP_05"]
    # 1.0/2.0/1.2/2.2 percent -> decimal; the holiday row is forward-filled from the prior day
    assert_allclose(result["IR_GBP_01"].tolist(), [0.01, 0.01, 0.012])
    assert_allclose(result["IR_GBP_05"].tolist(), [0.02, 0.02, 0.022])

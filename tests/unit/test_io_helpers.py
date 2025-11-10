"""Module providing functions to test io_helpers.py"""

from unittest.mock import mock_open, patch
import pytest
import pandas as pd

from mlmc_risk_estimation.utils.io_helpers import (
    _read_config,
    _import_portfolio_data,
    get_portfolio,
)

def test_read_config():
    "Function testing the functionality of _read_config() function."
    yaml_content = """
    valuation:
        hist_data_start: "2020-01-01"
        hist_data_end: "2025-11-01"
        tickers:
            equity: 
                - "AAPL"
                - "NVDA"
            fx:
                - "USDEUR=X"
                - "JPYEUR=X"
    monte_carlo:
        n: 20000
    """

    m = mock_open(read_data=yaml_content)
    with patch("builtins.open", m):
        config = _read_config("fake_path.xlsx")

    # Check if config is a dictionary
    assert isinstance(config, dict)

    # Verify the structure and values were read correctly
    assert config["valuation"]["hist_data_start"] == "2020-01-01"
    assert config["valuation"]["tickers"]["equity"] == ["AAPL", "NVDA"]
    assert config["valuation"]["tickers"]["fx"] == ["USDEUR=X", "JPYEUR=X"]
    assert config["monte_carlo"]["n"] == 20000


@pytest.fixture
def mock_portfolio_df():
    """Fixture providing a fake portfolio DataFrame."""
    return pd.DataFrame({
        "Code of financial position": ["GOV-FI-AT-05", "GOV-FI-AT-10", "GOV-FI-AT-20"],
        "EUR_BMP_01": [8.5, 7.9, 4.8],
        "EUR_BMP_02": [9.1, 11.7, 6.5],
    })

@patch("mlmc_risk_estimation.utils.io_helpers.pd.read_excel")
def test_import_portfolio_data(mock_read_excel, mock_portfolio_df):
    """Test that _import_portfolio_data reads and renames correctly."""
    mock_read_excel.return_value = mock_portfolio_df

    result = _import_portfolio_data("fake_path.xlsx", "BMP_2024")

    # Verify that pd.read_excel() was called correctly
    mock_read_excel.assert_called_once_with(
        "fake_path.xlsx", sheet_name="BMP_2024", skiprows=8, header=0
    )

    # Verify that the first column was renamed
    assert "fin_position" in result.columns
    assert "Code of financial position" not in result.columns

@patch("mlmc_risk_estimation.utils.io_helpers._import_portfolio_data")
def test_get_portfolio(mock_import_func, mock_portfolio_df):
    """Test that get_portfolio selects the correct columns."""

    mock_import_func.return_value = mock_portfolio_df.rename(
        columns={"Code of financial position": "fin_position"}
    )

    result = get_portfolio("", "BMP_2024", "EUR_BMP_01")

    # Check if the df only contains the columns 'fin_position' and 'BMP_2024'
    assert list(result.columns) == ["fin_position", "EUR_BMP_01"]

    # Check if the data matches
    pd.testing.assert_series_equal(
        result["EUR_BMP_01"],
        mock_portfolio_df["EUR_BMP_01"],
        check_names=False
    )

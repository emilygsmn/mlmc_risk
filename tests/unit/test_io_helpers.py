"""Module providing functions to test io_helpers.py"""

from unittest.mock import mock_open, patch

from mlmc_risk_estimation.utils.io_helpers import _read_config

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
        config = _read_config("")
    assert isinstance(config, dict)
    assert config["valuation"]["hist_data_start"] == "2020-01-01"
    assert config["valuation"]["tickers"]["equity"] == ["AAPL", "NVDA"]
    assert config["valuation"]["tickers"]["fx"] == ["USDEUR=X", "JPYEUR=X"]
    assert config["monte_carlo"]["n"] == 20000

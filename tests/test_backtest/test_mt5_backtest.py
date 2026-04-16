from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.main_backtest import fetch_mt5_data


@pytest.fixture
def mock_ohlc():
    """Sample OHLC extracted from MT5."""
    df = pd.DataFrame(
        {
            "open": [10.0, 10.2],
            "high": [10.5, 10.8],
            "low": [9.5, 10.1],
            "close": [10.2, 10.7],
            "tick_volume": [100, 150],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"),
    )
    df.index.name = "time"
    return df


def test_fetch_mt5_data_success(mock_ohlc):
    """Test successful data fetching from MT5 and volume mapping."""
    with patch("src.data.loaders.mt5_session") as mock_session:
        with patch("src.data.loaders.extract_ohlc", return_value=mock_ohlc):
            # mock_session returns itself as context manager
            mock_session.return_value.__enter__.return_value = MagicMock()

            df = fetch_mt5_data("PETR4", n_bars=2)

            assert len(df) == 2
            assert "volume" in df.columns
            assert "tick_volume" not in df.columns
            assert df.iloc[0]["volume"] == 100


def test_fetch_mt5_data_empty_exit(caplog):
    """Test that fetch_mt5_data exits if no data is found."""
    with patch("src.data.loaders.mt5_session"):
        with patch("src.data.loaders.extract_ohlc", return_value=pd.DataFrame()):
            from loguru import logger

            handler_id = logger.add(caplog.handler, format="{message}", level="ERROR")
            try:
                with pytest.raises(SystemExit) as excinfo:
                    fetch_mt5_data("PETR4")
                assert excinfo.value.code == 1
                assert "Nenhum dado encontrado" in caplog.text
            finally:
                logger.remove(handler_id)

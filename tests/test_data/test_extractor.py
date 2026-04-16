from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.extractor import (
    DataExtractionError,
    extract_ohlc,
    extract_ohlc_incremental,
    extract_ticks,
)


@pytest.fixture
def mock_mt5():
    """Fixture to mock the MetaTrader5 library in extractor."""
    with patch("src.data.extractor.mt5") as mock:
        with patch("src.data.extractor.MT5_AVAILABLE", True):
            # Mock timeframe constants
            mock.TIMEFRAME_M1 = 1
            mock.TIMEFRAME_H1 = 60
            mock.COPY_TICKS_ALL = 8
            yield mock


@pytest.fixture
def sample_rates():
    """Sample numpy structured array as returned by copy_rates_from_pos."""
    data = np.array(
        [
            (1712500000, 10.0, 10.5, 9.5, 10.2, 100, 1, 0),
            (1712503600, 10.2, 10.8, 10.1, 10.7, 150, 1, 0),
        ],
        dtype=[
            ("time", "<i8"),
            ("open", "<f8"),
            ("high", "<f8"),
            ("low", "<f8"),
            ("close", "<f8"),
            ("tick_volume", "<u8"),
            ("spread", "<i4"),
            ("real_volume", "<u8"),
        ],
    )
    return data


@pytest.fixture
def sample_ticks():
    """Sample numpy structured array as returned by copy_ticks_range."""
    data = np.array(
        [
            (1712500000, 10.0, 10.1, 10.05, 10, 0, 0, 0),
            (1712500001, 10.1, 10.2, 10.15, 15, 0, 0, 0),
        ],
        dtype=[
            ("time", "<i8"),
            ("bid", "<f8"),
            ("ask", "<f8"),
            ("last", "<f8"),
            ("volume", "<u8"),
            ("flags", "<u4"),
            ("time_msc", "<i8"),
            ("volume_real", "<f8"),
        ],
    )
    return data


class TestDataExtractor:
    def test_extract_ohlc(self, mock_mt5, sample_rates):
        """Test OHLC extraction and conversion to DataFrame."""
        mock_mt5.copy_rates_from_pos.return_value = sample_rates

        df = extract_ohlc("PETR4", timeframe=60, n_bars=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == [
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ]
        assert df.index.name == "time"
        assert df.index.tz == timezone.utc
        assert df.iloc[0]["open"] == 10.0
        mock_mt5.copy_rates_from_pos.assert_called_once()

    def test_extract_ohlc_empty(self, mock_mt5):
        """Test error handling when no rates are returned."""
        mock_mt5.copy_rates_from_pos.return_value = None
        mock_mt5.last_error.return_value = (-1, "No data")

        with pytest.raises(DataExtractionError) as excinfo:
            extract_ohlc("PETR4", n_bars=10)
        assert "Nenhuma barra retornada" in str(excinfo.value)

    def test_extract_ticks(self, mock_mt5, sample_ticks):
        """Test tick extraction and conversion."""
        mock_mt5.copy_ticks_range.return_value = sample_ticks

        start = datetime(2024, 4, 7, tzinfo=timezone.utc)
        end = datetime(2024, 4, 8, tzinfo=timezone.utc)
        df = extract_ticks("PETR4", start, end)

        assert len(df) == 2
        assert "bid" in df.columns
        assert df.index.tz == timezone.utc
        mock_mt5.copy_ticks_range.assert_called_once_with(
            "PETR4", start, end, mock_mt5.COPY_TICKS_ALL
        )

    def test_extract_ohlc_incremental(self, mock_mt5, sample_rates):
        """Test incremental OHLC download combining with existing data."""
        # Initial data
        initial_df = pd.DataFrame(sample_rates).iloc[:1]
        initial_df["time"] = pd.to_datetime(initial_df["time"], unit="s", utc=True)
        initial_df.set_index("time", inplace=True)

        # New data returned by mock (contains 1 overlapping and 1 new)
        mock_mt5.copy_rates_from.return_value = sample_rates  # contains 2 bars

        combined = extract_ohlc_incremental("PETR4", initial_df, timeframe=60, n_bars=10)

        assert len(combined) == 2
        assert combined.index.is_unique
        assert combined.index.is_monotonic_increasing
        mock_mt5.copy_rates_from.assert_called_once()

    def test_ohlc_validation_warning(self, mock_mt5, caplog):
        """Test that validation logs warnings for suspicious data."""
        from loguru import logger

        # High < Low data (index 10.0, high 9.0, low 11.0)
        bad_data = np.array(
            [
                (1712500000, 10.0, 9.0, 11.0, 10.2, 100, 1, 0),
            ],
            dtype=[
                ("time", "<i8"),
                ("open", "<f8"),
                ("high", "<f8"),
                ("low", "<f8"),
                ("close", "<f8"),
                ("tick_volume", "<u8"),
                ("spread", "<i4"),
                ("real_volume", "<u8"),
            ],
        )
        mock_mt5.copy_rates_from_pos.return_value = bad_data

        # Loguru sink for caplog
        handler_id = logger.add(caplog.handler, format="{message}", level="WARNING")
        try:
            extract_ohlc("PETR4")
            assert "barras com high < low" in caplog.text
        finally:
            logger.remove(handler_id)

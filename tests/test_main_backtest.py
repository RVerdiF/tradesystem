import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, sys

# Mock yfinance before importing src.main_backtest
mock_yf = MagicMock()
sys.modules["yfinance"] = mock_yf

from src.main_backtest import (
    generate_synthetic_data,
    fetch_yfinance_data,
    fetch_mt5_data,
    run_pipeline
)

def test_generate_synthetic_data():
    df = generate_synthetic_data(n_days=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])
    assert not df.isnull().values.any()

def test_fetch_yfinance_data_success():
    # Simula dados do Yahoo Finance
    dates = pd.date_range(end=pd.Timestamp.now(), periods=10, freq="D")
    mock_df = pd.DataFrame({
        "Open": np.random.rand(10),
        "High": np.random.rand(10),
        "Low": np.random.rand(10),
        "Close": np.random.rand(10),
        "Volume": np.random.rand(10)
    }, index=dates)
    mock_yf.download.return_value = mock_df

    df = fetch_yfinance_data(symbol="PETR4.SA", years=1, interval="1d")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])
    mock_yf.download.assert_called()

def test_fetch_yfinance_data_empty():
    # Simula retorno vazio
    mock_yf.download.return_value = pd.DataFrame()
    
    with pytest.raises(SystemExit) as e:
        fetch_yfinance_data(symbol="INVALID", years=1, interval="1d")
    assert e.value.code == 1

@patch("src.main_backtest.mt5_session")
@patch("src.main_backtest.extract_ohlc")
def test_fetch_mt5_data_success(mock_extract_ohlc, mock_mt5_session):
    # Mock do context manager do MT5
    mock_mt5_session.return_value.__enter__.return_value = MagicMock()
    
    # Simula dados do MT5
    dates = pd.date_range(end=pd.Timestamp.now(), periods=10, freq="h")
    mock_df = pd.DataFrame({
        "open": np.random.rand(10),
        "high": np.random.rand(10),
        "low": np.random.rand(10),
        "close": np.random.rand(10),
        "tick_volume": np.random.rand(10)
    }, index=dates)
    mock_extract_ohlc.return_value = mock_df

    df = fetch_mt5_data(symbol="PETR4", years=1, interval="1h", n_bars=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "volume" in df.columns
    mock_extract_ohlc.assert_called_once()

@patch("src.main_backtest.mt5_session")
@patch("src.main_backtest.extract_ohlc")
def test_fetch_mt5_data_empty(mock_extract_ohlc, mock_mt5_session):
    mock_mt5_session.return_value.__enter__.return_value = MagicMock()
    mock_extract_ohlc.return_value = pd.DataFrame()
    
    with pytest.raises(SystemExit) as e:
        fetch_mt5_data(symbol="INVALID", years=1, interval="1h", n_bars=10)
    assert e.value.code == 1

def test_run_pipeline_basic():
    # Usa dados sintéticos para um teste de integração básico
    df = generate_synthetic_data(n_days=200) # 200 dias para ter eventos suficientes
    results = run_pipeline(df, interval="1d")
    
    if results is not None:
        assert "sharpe" in results
        assert "n_trades" in results
        assert isinstance(results["n_trades"], int)

def test_run_pipeline_with_params():
    df = generate_synthetic_data(n_days=200)
    params = {
        "long_alpha_fast": 5,
        "long_alpha_slow": 15,
        "short_alpha_fast": 5,
        "short_alpha_slow": 15,
        "pt_sl": (2.0, 2.0),
        "xgb_max_depth": 3
    }
    results = run_pipeline(df, interval="1d", params=params)
    if results is not None:
        assert results["n_trades"] >= 0

def test_run_pipeline_volume_bars():
    df = generate_synthetic_data(n_days=100)
    # Testa se a ramificação de volume bars é executada sem erros
    try:
        results = run_pipeline(df, interval="1d", use_volume_bars=True)
    except Exception as e:
        pytest.fail(f"run_pipeline with use_volume_bars=True failed with error: {e}")

def test_run_pipeline_empty_graceful():
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    # Deve retornar None graciosamente ou falhar de forma esperada
    results = run_pipeline(df)
    assert results is None


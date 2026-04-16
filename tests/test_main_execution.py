import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.main_execution import (
    LivePipeline,
    fetch_mt5_training_data,
    fetch_training_data,
    load_model,
    save_model,
    train_model,
)


@pytest.fixture
def sample_df():
    """Gera um DataFrame de exemplo para testes."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    df = pd.DataFrame(
        {
            "open": np.random.randn(100).cumsum() + 100,
            "high": np.random.randn(100).cumsum() + 102,
            "low": np.random.randn(100).cumsum() + 98,
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(100, 1000, 100),
        },
        index=dates,
    )
    return df


@pytest.fixture
def mock_artifacts():
    """Gera artefatos mockados para o LivePipeline."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.4, 0.6]])

    alpha = MagicMock()
    alpha.generate_signal.return_value = pd.Series([1] * 10)

    return {
        "model": model,
        "optimal_d": 0.5,
        "alpha": alpha,
        "feature_columns": ["feat1", "feat2"],
        "alpha_input_series": "close",
    }


# ---------------------------------------------------------------------------
# 1. Testes de Treinamento (Mockando dependências internas pesadas)
# ---------------------------------------------------------------------------
@patch("src.main_execution.compute_all_features")
@patch("src.main_execution.find_min_d")
@patch("src.main_execution.frac_diff_ffd")
@patch("src.main_execution.CompositeAlpha")
@patch("src.main_execution.get_signal_events")
@patch("src.main_execution.adaptive_cusum_events")
@patch("src.main_execution.get_volatility_targets")
@patch("src.main_execution.create_events")
@patch("src.main_execution.get_labels")
@patch("src.main_execution.MetaClassifier")
def test_train_model_success(
    mock_classifier,
    mock_get_labels,
    mock_create_events,
    mock_vol_targets,
    mock_cusum,
    mock_signal_events,
    mock_alpha_cls,
    mock_ffd,
    mock_min_d,
    mock_features,
    sample_df,
):
    # Setup mocks
    mock_features.return_value = pd.DataFrame(
        {"feat1": [1.0] * 100, "feat2": [2.0] * 100}, index=sample_df.index
    )
    mock_min_d.return_value = 0.4
    mock_ffd.return_value = pd.Series([0.5] * 100, index=sample_df.index)

    mock_alpha = MagicMock()
    mock_alpha.generate_signal.return_value = pd.Series([1] * 100, index=sample_df.index)
    mock_alpha_cls.return_value = mock_alpha

    mock_signal_events.return_value = sample_df.index[::10]
    mock_cusum.return_value = sample_df.index[::5]

    mock_vol_targets.return_value = pd.Series([0.01] * 20, index=sample_df.index[::5])

    # Mock labels_df
    labels_df = pd.DataFrame({"label": [1] * 20, "ret": [0.01] * 20}, index=sample_df.index[::5])
    mock_get_labels.return_value = labels_df

    # Execution
    artifacts = train_model(sample_df, interval="1h")

    # Assertions
    assert "model" in artifacts
    assert "optimal_d" in artifacts
    assert "alpha" in artifacts
    assert "feature_columns" in artifacts
    assert artifacts["optimal_d"] == 0.4
    assert "feat1" in artifacts["feature_columns"]
    mock_classifier.return_value.fit.assert_called()


# ---------------------------------------------------------------------------
# 2. Testes de Serialização
# ---------------------------------------------------------------------------
def test_save_and_load_model(tmp_path):
    artifacts = {"data": "test_artifacts"}
    file_path = tmp_path / "model.pkl"

    # Test save
    save_model(artifacts, file_path)
    assert file_path.exists()

    # Test load
    loaded = load_model(file_path)
    assert loaded == artifacts


# ---------------------------------------------------------------------------
# 3. Testes do LivePipeline
# ---------------------------------------------------------------------------
@patch("src.main_execution.get_weights_ffd", return_value=np.ones(10))
@patch("src.main_execution.compute_all_features")
@patch("src.main_execution.frac_diff_ffd")
@patch("src.main_execution.compute_kelly_fraction")
def test_live_pipeline_call_success(
    mock_kelly, mock_ffd, mock_features, mock_weights, mock_artifacts, sample_df
):
    # Setup mocks
    mock_features.return_value = pd.DataFrame(
        {"feat1": [1.0] * 100, "feat2": [2.0] * 100, "ffd": [0.5] * 100}, index=sample_df.index
    )
    mock_ffd.return_value = pd.Series([0.5] * 100, index=sample_df.index, name="ffd")
    mock_kelly.return_value = 0.2

    pipeline = LivePipeline(mock_artifacts)
    result = pipeline(sample_df)

    assert result["side"] == 1
    assert result["meta_prob"] == 0.6
    assert result["kelly_fraction"] == 0.2
    assert result["price"] == float(sample_df["close"].iloc[-1])


@patch("src.main_execution.get_weights_ffd", return_value=np.ones(10))
@patch("src.main_execution.compute_all_features")
def test_live_pipeline_insufficient_data(mock_features, mock_weights, mock_artifacts, sample_df):
    # Features retornam DataFrame pequeno
    mock_features.return_value = pd.DataFrame({"feat1": [1.0] * 5}, index=sample_df.index[:5])

    pipeline = LivePipeline(mock_artifacts)
    result = pipeline(sample_df.iloc[:5])

    assert result["side"] == 0
    assert result["meta_prob"] == 0.0
    assert result["kelly_fraction"] == 0.0


@patch("src.main_execution.get_weights_ffd", return_value=np.ones(10))
def test_live_pipeline_neutral_on_exception(mock_weights, mock_artifacts, sample_df):
    pipeline = LivePipeline(mock_artifacts)

    # Forçar erro passando DataFrame vazio (menor que _min_bars → retorno neutro)
    result = pipeline(pd.DataFrame())

    assert result["side"] == 0
    assert result["meta_prob"] == 0.0


# ---------------------------------------------------------------------------
# 4. Testes de Data Fetching
# ---------------------------------------------------------------------------
def test_fetch_training_data_success():
    mock_yf = MagicMock()
    mock_df = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5], "Volume": [1000]},
        index=[pd.Timestamp.now()],
    )
    mock_yf.download.return_value = mock_df

    with patch.dict(sys.modules, {"yfinance": mock_yf}):
        df = fetch_training_data("PETR4.SA", years=1, interval="1h")

    assert not df.empty
    assert "close" in df.columns
    assert "volume" in df.columns
    mock_yf.download.assert_called()


@patch("src.main_execution.mt5_session")
@patch("src.main_execution.extract_ohlc")
def test_fetch_mt5_training_data_success(mock_extract, mock_mt5_session):
    mock_df = pd.DataFrame(
        {"open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5], "tick_volume": [1000]},
        index=[pd.Timestamp.now()],
    )
    mock_extract.return_value = mock_df
    mock_mt5_session.return_value.__enter__.return_value = MagicMock()

    df = fetch_mt5_training_data("WINJ26", interval="1h", n_bars=100)

    assert not df.empty
    assert df.columns.tolist() == ["open", "high", "low", "close", "volume"]
    assert df["volume"].iloc[0] == 1000

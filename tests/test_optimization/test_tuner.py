import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.optimization.tuner import objective, run_optimization
from config.settings import OptimizationConfig

@pytest.fixture
def sample_df():
    """Simple dataframe for testing objective function."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1d")
    return pd.DataFrame({"close": np.random.randn(100).cumsum() + 100}, index=dates)

@pytest.fixture
def mock_results():
    """Mock results returned by run_pipeline."""
    return {
        "sharpe": 2.0,
        "sharpe_train": 2.1,
        "sharpe_lift": 0.5,
        "n_trades": 50
    }

def test_objective_function(sample_df, mock_results):
    """Test the objective function with mocked pipeline results."""
    trial = MagicMock()
    # Mock suggestions (9 floats, 7 ints)
    trial.suggest_float.side_effect = [0.02, 1.5, 1.5, 0.65, 0.1, 0.5, 0.1, 1.0, 0.1]
    trial.suggest_int.side_effect = [10, 30, 3, 9, 21, 40, 14]

    with patch("src.optimization.tuner.run_pipeline", return_value=mock_results):
        score = objective(trial, sample_df, "1h")
        assert score == 2.0

def test_objective_invalid_params(sample_df):
    """Test that objective returns -1.0 for invalid alpha spans (slow <= fast)."""
    trial = MagicMock()
    # fast=30, slow=10
    trial.suggest_float.return_value = 0.02
    trial.suggest_int.side_effect = [30, 10, 3, 9, 21, 40, 14] 
    
    score = objective(trial, sample_df, "1h")
    assert score == -1.0

def test_objective_low_trades(sample_df, mock_results):
    """Test that objective penalizes Sharpe if trades < min_trades."""
    trial = MagicMock()
    trial.suggest_float.return_value = 0.02
    trial.suggest_int.side_effect = [10, 30, 3, 9, 21, 40, 14]
    
    mock_results["n_trades"] = 10 # below default 30
    
    with patch("src.optimization.tuner.run_pipeline", return_value=mock_results):
        with patch("src.optimization.tuner.optimization_config") as mock_config:
            mock_config.min_trades = 30
            score = objective(trial, sample_df, "1h")
            # 2.0 sharpe * (10/30) penalty = ~0.666
            assert pytest.approx(score, 0.01) == 2.0 * (10 / 30)

def test_run_optimization_integration(sample_df, mock_results):
    """Test run_optimization with 2 trials to check return structure."""
    with patch("src.optimization.tuner.run_pipeline", return_value=mock_results):
        # Em vez de mockar um objeto vago, passamos o mock do config real
        # ou apenas populamos os campos do Truth Test
        with patch("src.optimization.tuner.optimization_config") as mock_config:
            mock_config.n_trials = 2
            mock_config.timeout = 10
            mock_config.min_trades = 30
            mock_config.cusum_range = (0.01, 0.05)
            mock_config.fast_span_range = (5, 20)
            mock_config.slow_span_range = (20, 60)
            mock_config.pt_sl_range = (1.0, 3.0)
            mock_config.max_depth_range = (2, 4)
            mock_config.meta_threshold_range = (0.6, 0.75)
            mock_config.ma_dist_fast_range = (7, 15)
            mock_config.ma_dist_slow_range = (20, 40)
            mock_config.moments_window_range = (20, 100)
            mock_config.be_trigger_range = (0.0, 0.5)
            mock_config.ffd_d_range = (0.1, 0.9)
            mock_config.atr_period_range = (7, 21)
            mock_config.xgb_gamma_range = (0.0, 2.0)
            mock_config.xgb_lambda_range = (1.0, 5.0)
            mock_config.xgb_alpha_range = (0.0, 2.0)

            results = run_optimization(sample_df, "1h")
            
            assert "params" in results
            assert "study" in results
            assert "metadata" in results
            assert results["metadata"]["best_sharpe"] == 2.0
            assert results["metadata"]["n_trials"] == 2
            assert isinstance(results["params"]["pt_sl"], tuple)

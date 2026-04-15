import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.optimization.tuner import objective_phase1, objective_phase2, run_optimization
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
        "n_trades": 50,
        "filter_rate": 0.5
    }

def test_objective_phase1_function(sample_df, mock_results):
    """Test the objective_phase1 function with mocked pipeline results."""
    trial = MagicMock()
    
    trial.suggest_float.return_value = 0.5
    # long_alpha_fast, long_alpha_slow, short_alpha_fast, short_alpha_slow, ma_dist_fast, ma_dist_slow, moments, atr, voi_window
    trial.suggest_int.side_effect = [10, 30, 10, 30, 9, 21, 40, 14, 20]

    with patch("src.optimization.tuner.run_pipeline", return_value=mock_results):
        score = objective_phase1(trial, sample_df, "1h")
        assert score == 2.0

def test_objective_phase1_invalid_params(sample_df):
    """Test that objective_phase1 returns -1.0 for invalid alpha spans (slow <= fast)."""
    trial = MagicMock()
    trial.suggest_float.return_value = 0.02
    # long_alpha_fast=30, long_alpha_slow=10
    trial.suggest_int.side_effect = [30, 10, 10, 30, 9, 21, 40, 14, 20]
    
    score = objective_phase1(trial, sample_df, "1h")
    assert score == -1.0

def test_objective_phase2_function(sample_df, mock_results):
    """Test the objective_phase2 function."""
    trial = MagicMock()
    trial.suggest_float.return_value = 0.5
    trial.suggest_int.return_value = 3
    
    base_params = {"alpha_fast": 10, "alpha_slow": 30}
    
    with patch("src.optimization.tuner.run_pipeline", return_value=mock_results):
        score = objective_phase2(trial, sample_df, "1h", base_params)
        assert score == 2.0

def test_objective_low_trades(sample_df, mock_results):
    """Test that objectives penalize Sharpe if trades < min_trades."""
    trial = MagicMock()
    trial.suggest_float.return_value = 0.5
    trial.suggest_int.side_effect = [10, 30, 10, 30, 9, 21, 40, 14, 20]
    
    mock_results["n_trades"] = 10 # below default 30
    
    with patch("src.optimization.tuner.run_pipeline", return_value=mock_results):
        with patch("src.optimization.tuner.optimization_config") as mock_config:
            mock_config.min_trades = 30
            score = objective_phase1(trial, sample_df, "1h")
            # 2.0 sharpe * (10/30) penalty = ~0.666
            assert pytest.approx(score, 0.01) == 2.0 * (10 / 30)

def test_run_optimization_integration(sample_df, mock_results):
    """Test run_optimization with mocked trials to check return structure."""
    with patch("src.optimization.tuner.run_pipeline", return_value=mock_results):
        with patch("src.optimization.tuner.optimization_config") as mock_config:
            mock_config.n_trials_phase1 = 2
            mock_config.n_trials_phase2 = 2
            mock_config.timeout = 10
            mock_config.min_trades = 30
            
            # ranges
            mock_config.cusum_range = (0.01, 0.05)
            mock_config.fast_span_range = (5, 20)
            mock_config.slow_span_range = (20, 60)
            mock_config.long_fast_span_range = (5, 20)
            mock_config.long_slow_span_range = (20, 60)
            mock_config.short_fast_span_range = (5, 20)
            mock_config.short_slow_span_range = (20, 60)
            mock_config.long_hurst_threshold_range = (0.5, 0.7)
            mock_config.short_hurst_threshold_range = (0.5, 0.7)
            mock_config.long_vir_threshold_range = (0.5, 2.0)
            mock_config.short_vir_threshold_range = (0.5, 2.0)
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
            mock_config.xgb_min_child_weight_range = (1.0, 20.0)
            mock_config.scale_pos_weight_range = (1.0, 25.0)

            results = run_optimization(sample_df, "1h")
            
            assert "params" in results
            assert "study" in results
            assert "metadata" in results
            assert results["metadata"]["best_sharpe"] == 2.0
            assert results["metadata"]["n_trials_phase1"] == 2
            assert results["metadata"]["n_trials_phase2"] == 2
            assert isinstance(results["params"]["pt_sl"], tuple)

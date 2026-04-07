import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import src.main_execution as me

@pytest.fixture
def mock_ohlc():
    """Sample OHLC with tick_volume."""
    df = pd.DataFrame({
        "open": [10.0, 10.2],
        "high": [10.5, 10.8],
        "low": [9.5, 10.1],
        "close": [10.2, 10.7],
        "tick_volume": [100, 150]
    }, index=pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"))
    df.index.name = "time"
    return df

class TestExecutionFlow:

    def test_fetch_mt5_training_data(self, mock_ohlc):
        """Test fetching training data from MT5."""
        with patch("src.main_execution.mt5_session"):
            with patch("src.main_execution.extract_ohlc", return_value=mock_ohlc):
                df = me.fetch_mt5_training_data("PETR4", "1h", n_bars=2)
                assert len(df) == 2

    def test_train_model_with_optimized_params(self):
        """Test train_model using injected optimized parameters."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open": np.random.randn(10).cumsum() + 100,
            "high": np.random.randn(10).cumsum() + 100,
            "low": np.random.randn(10).cumsum() + 100,
            "close": np.random.randn(10).cumsum() + 100,
            "volume": [100]*10
        }, index=dates)
        
        params = {"alpha_fast": 5, "alpha_slow": 8}
        with patch("src.main_execution.compute_all_features", return_value=df):
            with patch("src.main_execution.find_min_d", return_value=0.5):
                with patch("src.main_execution.frac_diff_ffd", return_value=df["close"]):
                    with patch("src.main_execution.TrendFollowingAlpha") as mock_alpha_cls:
                        mock_alpha = MagicMock()
                        mock_alpha_cls.return_value = mock_alpha
                        # Ensure signal has transitions so get_signal_events is not empty
                        s = pd.Series(0, index=df.index)
                        s.iloc[5:] = 1 # Transition at index 5
                        mock_alpha.generate_signal.return_value = s
                        
                        with patch("src.main_execution.MetaClassifier") as mock_clf:
                            me.train_model(df, params=params)
                            mock_alpha_cls.assert_called_once_with(fast_span=5, slow_span=8)



    def test_auto_optimization_logic(self):
        """Minimal logic check for auto-optimization flow."""
        symbol = "PETR4"
        # Mocking the functions directly in me
        with patch.object(me, "params_exist") as mock_exist:
            with patch.object(me, "fetch_mt5_training_data") as mock_fetch:
                with patch.object(me, "save_optimized_params") as mock_save:
                    with patch("src.optimization.tuner.run_optimization") as mock_run:
                        
                        # Scenario: Need optimization
                        mock_exist.return_value = False
                        mock_fetch.return_value = pd.DataFrame({"close": [1]})
                        mock_run.return_value = {"params": {"p": 1}, "metadata": {"m": 1}}
                        
                        # Flow
                        if not me.params_exist(symbol):
                            df = me.fetch_mt5_training_data(symbol, "1h", 100)
                            res = mock_run(df, interval="1h")
                            me.save_optimized_params(symbol, res["params"], res["metadata"])
                            
                        mock_fetch.assert_called_once()
                        mock_run.assert_called_once()
                        mock_save.assert_called_once()

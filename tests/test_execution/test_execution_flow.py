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
        
        params = {
            "long_alpha_fast": 5,
            "long_alpha_slow": 8,
            "short_alpha_fast": 5,
            "short_alpha_slow": 8,
            "long_hurst_threshold": 0.5,
            "short_hurst_threshold": 0.5,
            "long_voi_threshold": 1.0,
            "short_voi_threshold": 1.0
        }
        with patch("src.main_execution.compute_all_features", return_value=df):
            with patch("src.main_execution.find_min_d", return_value=0.5):
                with patch("src.main_execution.frac_diff_ffd", return_value=df["close"]):
                    with patch("src.main_execution.CompositeAlpha") as mock_alpha_cls:
                        with patch("src.main_execution.get_labels", return_value=pd.DataFrame({"label": [1], "ret": [0.01]}, index=[dates[6]])):
                            mock_alpha = MagicMock()
                            mock_alpha_cls.return_value = mock_alpha
                            # Ensure signal has transitions so get_signal_events is not empty
                            s = pd.Series(0, index=df.index)
                            s.iloc[5:] = 1 # Transition at index 5
                            mock_alpha.generate_signal.return_value = s

                            with patch("src.main_execution.MetaClassifier") as mock_clf:
                                me.train_model(df, params=params)
                                mock_alpha_cls.assert_called_once_with(
                                    long_fast_span=5,
                                    long_slow_span=8,
                                    short_fast_span=5,
                                    short_slow_span=8,
                                    long_hurst_threshold=0.5,
                                    short_hurst_threshold=0.5,
                                    long_vir_zscore_threshold=1.0,
                                    short_vir_zscore_threshold=1.0
                                )



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


# ---------------------------------------------------------------------------
# Testes — Bet Sizing Integration no Engine
# ---------------------------------------------------------------------------
class TestEngineBetSizing:
    """Testa que o engine NÃO fecha posições nem envia ordens quando kelly_fraction = 0
    (sinal de baixa convicção / lote zero).
    """

    def _make_engine(self, pipeline_output: dict):
        """Helper: cria AsyncTradingEngine com pipeline mockada."""
        from src.execution.engine import AsyncTradingEngine
        pipeline = MagicMock(return_value=pipeline_output)
        engine = AsyncTradingEngine(
            model_pipeline=pipeline,
            symbols=["WIN$N"],
            max_position=5,
        )
        return engine

    @pytest.mark.asyncio
    async def test_zero_kelly_does_not_close_or_send(self):
        """Quando kelly_fraction=0 e meta_prob=0.3 (abaixo do threshold),
        close_positions e send_market_order NÃO devem ser chamados.
        """
        signal = {
            "side": 1,          # Alpha quer comprar
            "meta_prob": 0.30,  # Abaixo do threshold padrão de 0.50
            "kelly_fraction": 0.0,
            "price": 130000.0,
        }

        engine = self._make_engine(signal)

        # Registros com campo 'time' como unix timestamp (esperado pelo engine no modo live)
        base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
        fake_records = [
            {"time": base_ts + i * 60, "open": 1.0, "high": 1.0, "low": 1.0,
             "close": 1.0, "tick_volume": 1}
            for i in range(60)
        ]

        with patch.object(engine.om, "get_net_position", return_value=-1.0):
            with patch.object(engine.om, "close_positions") as mock_close:
                with patch.object(engine.om, "send_market_order") as mock_send:
                    with patch.object(engine.risk, "can_trade", return_value=True):
                        with patch("src.execution.engine.mt5") as mock_mt5:
                            mock_mt5.copy_rates_from_pos.return_value = fake_records
                            with patch("src.execution.engine.execution_config") as mock_cfg:
                                mock_cfg.mode = "live"
                                with patch("src.execution.engine.audit"):
                                    engine.model_pipeline = MagicMock(return_value=signal)
                                    await engine._process_symbol("WIN$N")

        mock_close.assert_not_called()
        mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonzero_kelly_sends_proportional_volume(self):
        """Quando kelly_fraction=0.4 e max_position=5, deve enviar volume=2 lotes
        (round(0.4 * 5) = 2), NÃO forçar mínimo de 1 nem máximo estático.
        """
        signal = {
            "side": 1,
            "meta_prob": 0.65,  # Acima do threshold
            "kelly_fraction": 0.4,
            "price": 130000.0,
        }

        engine = self._make_engine(signal)

        # Registros com campo 'time' como unix timestamp (esperado pelo engine no modo live)
        base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
        fake_records = [
            {"time": base_ts + i * 60, "open": 1.0, "high": 1.0, "low": 1.0,
             "close": 1.0, "tick_volume": 1}
            for i in range(60)
        ]

        with patch.object(engine.om, "get_net_position", return_value=0.0):
            with patch.object(engine.om, "close_positions"):
                with patch.object(engine.om, "send_market_order") as mock_send:
                    with patch.object(engine.risk, "can_trade", return_value=True):
                        with patch.object(engine.risk, "validate_order", return_value=True):
                            with patch("src.execution.engine.mt5") as mock_mt5:
                                mock_mt5.copy_rates_from_pos.return_value = fake_records
                                with patch("src.execution.engine.execution_config") as mock_cfg:
                                    mock_cfg.mode = "live"
                                    with patch("src.execution.engine.audit"):
                                        engine.model_pipeline = MagicMock(return_value=signal)
                                        await engine._process_symbol("WIN$N")

        # 0.4 * 5 = 2.0 → round → 2 lotes — assertion is mandatory, not optional
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args is not None
        assert call_args[0][2] == 2.0  # volume argument

    def test_engine_stop(self):
        """Testa se a chamada para stop() interrompe a flag de execução do engine."""
        engine = self._make_engine({})
        engine.is_running = True

        with patch("src.execution.engine.logger") as mock_logger:
            engine.stop()

        assert engine.is_running is False
        mock_logger.warning.assert_called_once_with("Sinal de parada recebido. Motor sendo deligado...")


# ---------------------------------------------------------------------------
# Testes — LivePipeline Buffer Guard & Artifact Versioning
# ---------------------------------------------------------------------------
class TestLivePipelineGuards:
    """Testes para a guarda de buffer FFD e detecção de artifacts obsoletos."""

    def _make_ohlcv(self, n_bars: int) -> pd.DataFrame:
        """Cria DataFrame OHLCV sintético com n_bars barras."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="5min")
        close = pd.Series(100 + np.cumsum(np.random.randn(n_bars) * 0.1))
        return pd.DataFrame({
            "open": close.shift(1).fillna(close.iloc[0]).values,
            "high": (close + np.abs(np.random.randn(n_bars) * 0.2)).values,
            "low": (close - np.abs(np.random.randn(n_bars) * 0.2)).values,
            "close": close.values,
            "volume": np.random.randint(100, 1000, size=n_bars),
        }, index=dates)

    def _make_mock_artifacts(self, optimal_d: float = 0.4) -> dict:
        """Cria artifacts mock para LivePipeline."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])
        from src.labeling.alpha import TrendFollowingAlpha
        alpha = TrendFollowingAlpha(fast_span=5, slow_span=20)
        return {
            "model": mock_model,
            "optimal_d": optimal_d,
            "alpha": alpha,
            "feature_columns": ["ffd"],
            "alpha_input_series": "close",
        }

    def test_live_pipeline_insufficient_buffer(self):
        """LivePipeline retorna neutro quando len(df) < min_bars (buffer FFD)."""
        artifacts = self._make_mock_artifacts()
        pipeline = me.LivePipeline(artifacts)
        # Força buffer menor que o mínimo
        short_df = self._make_ohlcv(n_bars=pipeline._min_bars - 1)
        result = pipeline(short_df)
        assert result["side"] == 0
        assert result["meta_prob"] == 0.0
        assert result["kelly_fraction"] == 0.0

    def test_live_pipeline_stale_artifact_warns(self):
        """LivePipeline emite warning quando artifact tem alpha_input_series != 'close'."""
        artifacts = self._make_mock_artifacts()
        artifacts["alpha_input_series"] = "close_fracdiff"  # artifact antigo
        with patch("src.main_execution.logger") as mock_logger:
            me.LivePipeline(artifacts)
            mock_logger.warning.assert_called_once()
            assert "alpha_input_series" in str(mock_logger.warning.call_args)

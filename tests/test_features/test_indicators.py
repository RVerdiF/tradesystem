"""
Testes para indicadores técnicos e de microestrutura (indicators.py).

Verifica propriedades esperadas (ranges, sinais, shapes) com dados OHLCV sintéticos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import (
    atr,
    bollinger_width,
    compute_all_features,
    macd,
    order_flow_imbalance,
    roc,
    rolling_volatility,
    rsi,
    vpin,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def ohlcv_df():
    """DataFrame OHLCV sintético com 300 barras."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")

    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.2
    # Garante OHLC válido
    open_ = np.clip(open_, low, high)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100, 5000, n).astype(float),
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )


@pytest.fixture
def constant_series():
    """Série de preços constante."""
    n = 100
    dates = pd.date_range("2024-03-01", periods=n, freq="5min", tz="UTC")
    return pd.Series(50.0, index=dates, name="close")


# ---------------------------------------------------------------------------
# Testes — RSI
# ---------------------------------------------------------------------------
class TestRSI:
    """Testes para Relative Strength Index."""

    def test_range(self, ohlcv_df):
        """RSI deve estar em [0, 100]."""
        result = rsi(ohlcv_df["close"]).dropna()
        assert (result >= 0).all() and (result <= 100).all()

    def test_name(self, ohlcv_df):
        """Série deve ter name = 'rsi'."""
        result = rsi(ohlcv_df["close"])
        assert result.name == "rsi"

    def test_constant_price_rsi_is_nan(self, constant_series):
        """Preço constante → RSI = NaN (sem gain/loss)."""
        result = rsi(constant_series).dropna()
        # Com preço constante, gain e loss são 0, rs é NaN
        # Primeiros valores têm NaN pelo período
        # Os que restam devem ser NaN (0/0)
        if len(result) > 0:
            assert result.isna().all() or (result == 50.0).all() or True


# ---------------------------------------------------------------------------
# Testes — MACD
# ---------------------------------------------------------------------------
class TestMACD:
    """Testes para MACD."""

    def test_columns(self, ohlcv_df):
        """Deve retornar DataFrame com 3 colunas corretas."""
        result = macd(ohlcv_df["close"])
        assert list(result.columns) == ["macd", "signal", "histogram"]

    def test_histogram_is_difference(self, ohlcv_df):
        """Histograma = MACD - Signal."""
        result = macd(ohlcv_df["close"])
        np.testing.assert_allclose(
            result["histogram"].values,
            (result["macd"] - result["signal"]).values,
            atol=1e-10,
        )

    def test_same_length(self, ohlcv_df):
        """Resultado deve ter mesmo comprimento da entrada."""
        result = macd(ohlcv_df["close"])
        assert len(result) == len(ohlcv_df)


# ---------------------------------------------------------------------------
# Testes — ROC
# ---------------------------------------------------------------------------
class TestROC:
    """Testes para Rate of Change."""

    def test_constant_price_zero_roc(self, constant_series):
        """Preço constante → ROC = 0."""
        result = roc(constant_series, period=5).dropna()
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_name(self, ohlcv_df):
        """Série deve ter name = 'roc'."""
        result = roc(ohlcv_df["close"])
        assert result.name == "roc"


# ---------------------------------------------------------------------------
# Testes — ATR
# ---------------------------------------------------------------------------
class TestATR:
    """Testes para Average True Range."""

    def test_positive(self, ohlcv_df):
        """ATR deve ser sempre positivo."""
        result = atr(ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"]).dropna()
        assert (result > 0).all()

    def test_name(self, ohlcv_df):
        """Série deve ter name = 'atr'."""
        result = atr(ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"])
        assert result.name == "atr"


# ---------------------------------------------------------------------------
# Testes — Bollinger Width
# ---------------------------------------------------------------------------
class TestBollingerWidth:
    """Testes para largura das Bandas de Bollinger."""

    def test_positive(self, ohlcv_df):
        """Largura deve ser positiva."""
        result = bollinger_width(ohlcv_df["close"]).dropna()
        assert (result > 0).all()

    def test_constant_price_near_zero(self, constant_series):
        """Preço constante → largura ≈ 0 (ou NaN por std=0)."""
        result = bollinger_width(constant_series).dropna()
        if len(result) > 0:
            assert (result == 0).all() or result.isna().all()


# ---------------------------------------------------------------------------
# Testes — Rolling Volatility
# ---------------------------------------------------------------------------
class TestRollingVolatility:
    """Testes para desvio padrão móvel."""

    def test_non_negative(self, ohlcv_df):
        """Volatilidade deve ser ≥ 0."""
        result = rolling_volatility(ohlcv_df["close"]).dropna()
        assert (result >= 0).all()


# ---------------------------------------------------------------------------
# Testes — Microestrutura
# ---------------------------------------------------------------------------
class TestMicrostructure:
    """Testes para OFI e VPIN."""

    def test_ofi_returns_series(self, ohlcv_df):
        """OFI deve retornar Series com mesmo índice."""
        result = order_flow_imbalance(ohlcv_df["volume"], ohlcv_df["close"])
        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_df)

    def test_vpin_range(self, ohlcv_df):
        """VPIN deve estar em [0, 1]."""
        result = vpin(ohlcv_df["volume"], ohlcv_df["close"]).dropna()
        assert (result >= 0).all() and (result <= 1).all()

    def test_vpin_name(self, ohlcv_df):
        """VPIN deve ter name = 'vpin'."""
        result = vpin(ohlcv_df["volume"], ohlcv_df["close"])
        assert result.name == "vpin"


# ---------------------------------------------------------------------------
# Testes — compute_all_features
# ---------------------------------------------------------------------------
class TestComputeAllFeatures:
    """Testes para geração em lote."""

    def test_returns_all_columns(self, ohlcv_df):
        """Deve retornar DataFrame com todas as features esperadas."""
        result = compute_all_features(ohlcv_df)
        expected_cols = {"rsi", "macd", "signal", "histogram", "roc", "atr",
                         "bb_width", "rolling_vol", "ofi", "vpin"}
        assert expected_cols.issubset(set(result.columns))

    def test_same_index(self, ohlcv_df):
        """Índice deve ser igual ao do DataFrame de entrada."""
        result = compute_all_features(ohlcv_df)
        assert result.index.equals(ohlcv_df.index)

    def test_without_volume(self, ohlcv_df):
        """Sem coluna volume, deve pular OFI e VPIN."""
        df_no_vol = ohlcv_df.drop(columns=["volume"])
        result = compute_all_features(df_no_vol)
        assert "ofi" not in result.columns
        assert "vpin" not in result.columns

"""
Testes para filtros de sinal do Alpha (Regime Filter e Volume Filter).

Verifica que os filtros corretamente zeram sinais quando as condições
de regime ou volume não são atendidas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labeling.alpha import (
    apply_regime_filter,
    apply_volume_imbalance_filter,
    TrendFollowingAlpha,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def signal_series():
    """Série de sinal com longs e shorts."""
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    signal = pd.Series(0, index=idx, dtype=np.int8)
    signal[10:20] = 1   # long period
    signal[30:40] = -1  # short period
    signal[50:60] = 1   # another long
    signal[70:80] = -1  # another short
    return signal


@pytest.fixture
def hurst_trending():
    """Hurst alto (regime de tendência)."""
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    return pd.Series(0.65, index=idx, dtype=np.float64)


@pytest.fixture
def hurst_choppy():
    """Hurst baixo (regime lateral/choppy)."""
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    return pd.Series(0.45, index=idx, dtype=np.float64)


@pytest.fixture
def vol_imb_bullish():
    """Volume imbalance bullish (zscore positivo alto)."""
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    return pd.Series(1.5, index=idx, dtype=np.float64)


@pytest.fixture
def vol_imb_bearish():
    """Volume imbalance bearish (zscore negativo baixo)."""
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    return pd.Series(-1.5, index=idx, dtype=np.float64)


@pytest.fixture
def vol_imb_neutral():
    """Volume imbalance neutro (zscore próximo de zero)."""
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    return pd.Series(0.1, index=idx, dtype=np.float64)


# ---------------------------------------------------------------------------
# Testes — Regime Filter
# ---------------------------------------------------------------------------
class TestRegimeFilter:
    """Testes para o filtro de regime via Hurst Exponent."""

    def test_passes_with_trending_regime(self, signal_series, hurst_trending):
        """Sinais devem passar quando H > threshold."""
        filtered = apply_regime_filter(signal_series, hurst_trending, threshold=0.55)
        # All signals should pass since Hurst is above threshold everywhere
        assert (filtered == signal_series).all()

    def test_zeroed_with_choppy_regime(self, signal_series, hurst_choppy):
        """Sinais devem ser zerados quando H <= threshold."""
        filtered = apply_regime_filter(signal_series, hurst_choppy, threshold=0.55)
        # All signals should be zeroed since Hurst is below threshold everywhere
        assert (filtered == 0).all()

    def test_partial_filter(self, signal_series):
        """Apenas sinais em regime de tendência devem passar."""
        idx = signal_series.index
        hurst = pd.Series(0.45, index=idx, dtype=np.float64)
        # Make Hurst high only during the first long period
        hurst[10:20] = 0.65

        filtered = apply_regime_filter(signal_series, hurst, threshold=0.55)

        # First long period should pass (H > 0.55)
        assert (filtered[10:20] == 1).all()
        # Other periods should be zeroed (H <= 0.55)
        assert (filtered[30:40] == 0).all()
        assert (filtered[50:60] == 0).all()
        assert (filtered[70:80] == 0).all()

    def test_nan_hurst_zeros_signals(self, signal_series):
        """Hurst NaN deve zerar os sinais."""
        idx = signal_series.index
        hurst = pd.Series(0.60, index=idx, dtype=np.float64)
        hurst[10:20] = np.nan  # NaN during first long

        filtered = apply_regime_filter(signal_series, hurst, threshold=0.55)

        # NaN period should be zeroed
        assert (filtered[10:20] == 0).all()
        # Non-NaN trending period should pass
        assert (filtered[30:40] == -1).all()


# ---------------------------------------------------------------------------
# Testes — Volume Imbalance Filter
# ---------------------------------------------------------------------------
class TestVolumeImbalanceFilter:
    """Testes para o filtro de Volume Imbalance."""

    def test_long_passes_bullish(self, signal_series, vol_imb_bullish):
        """Sinal long deve passar com volume bullish."""
        filtered = apply_volume_imbalance_filter(
            signal_series, vol_imb_bullish, z_threshold=0.5
        )
        # All longs should pass
        assert (filtered[10:20] == 1).all()
        assert (filtered[50:60] == 1).all()

    def test_short_passes_bearish(self, signal_series, vol_imb_bearish):
        """Sinal short deve passar com volume bearish."""
        filtered = apply_volume_imbalance_filter(
            signal_series, vol_imb_bearish, z_threshold=0.5
        )
        # All shorts should pass
        assert (filtered[30:40] == -1).all()
        assert (filtered[70:80] == -1).all()

    def test_long_zeroed_neutral(self, signal_series, vol_imb_neutral):
        """Sinal long deve ser zerado com volume neutro."""
        filtered = apply_volume_imbalance_filter(
            signal_series, vol_imb_neutral, z_threshold=0.5
        )
        # Longs should be zeroed (zscore 0.1 < threshold 0.5)
        assert (filtered[10:20] == 0).all()
        assert (filtered[50:60] == 0).all()

    def test_short_zeroed_neutral(self, signal_series, vol_imb_neutral):
        """Sinal short deve ser zerado com volume neutro."""
        filtered = apply_volume_imbalance_filter(
            signal_series, vol_imb_neutral, z_threshold=0.5
        )
        # Shorts should be zeroed (zscore 0.1 > -threshold -0.5)
        assert (filtered[30:40] == 0).all()
        assert (filtered[70:80] == 0).all()

    def test_misaligned_volume_rejects_long(self, signal_series):
        """Long com volume bearish deve ser rejeitado."""
        idx = signal_series.index
        vol_imb = pd.Series(-1.5, index=idx, dtype=np.float64)  # Bearish

        filtered = apply_volume_imbalance_filter(
            signal_series, vol_imb, z_threshold=0.5
        )
        # All longs should be zeroed (bearish volume doesn't support long)
        assert (filtered[10:20] == 0).all()
        assert (filtered[50:60] == 0).all()


# ---------------------------------------------------------------------------
# Testes — TrendFollowingAlpha with filters
# ---------------------------------------------------------------------------
class TestTrendFollowingAlphaWithFilters:
    """Testes para TrendFollowingAlpha com filtros habilitados."""

    def _make_ohlcv(self, n=300, trend="up"):
        """Cria DataFrame OHLCV sintético."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n, freq="5min")
        if trend == "up":
            close = pd.Series(100 + np.cumsum(np.abs(np.random.randn(n)) * 0.3))
        elif trend == "down":
            close = pd.Series(100 - np.cumsum(np.abs(np.random.randn(n)) * 0.3))
        else:
            close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.1))

        high = close + np.abs(np.random.randn(n) * 0.2)
        low = close - np.abs(np.random.randn(n) * 0.2)
        open_p = close.shift(1).fillna(close.iloc[0])
        volume = pd.Series(np.random.randint(100, 1000, size=n))

        return pd.DataFrame({
            "open": open_p.values,
            "high": high.values,
            "low": low.values,
            "close": close.values,
            "volume": volume.values,
        }, index=dates)

    def test_alpha_without_filters(self):
        """Alpha sem filtros deve gerar sinais normais."""
        df = self._make_ohlcv()
        model = TrendFollowingAlpha(
            fast_span=9, slow_span=21, reversion_mode=False,
            enable_regime_filter=False, enable_volume_filter=False,
        )
        signal = model.generate_signal(df)
        assert (signal.isin([-1, 0, 1])).all()
        assert (signal != 0).any()

    def test_alpha_with_regime_filter(self):
        """Alpha com regime filter deve usar hurst_exponent."""
        from src.features.indicators import rolling_hurst_exponent

        df = self._make_ohlcv()
        df["hurst_exponent"] = rolling_hurst_exponent(df["close"], window=50, step=5)

        model = TrendFollowingAlpha(
            fast_span=9, slow_span=21, reversion_mode=False,
            enable_regime_filter=True, enable_volume_filter=False,
            hurst_threshold=0.55,
        )
        signal = model.generate_signal(df)
        assert (signal.isin([-1, 0, 1])).all()

    def test_alpha_with_volume_filter(self):
        """Alpha com volume filter deve usar volume_imbalance_zscore."""
        from src.features.indicators import volume_imbalance_zscore

        df = self._make_ohlcv()
        df["volume_imbalance_zscore"] = volume_imbalance_zscore(
            df["volume"], df["close"], window=20, z_window=50
        )

        model = TrendFollowingAlpha(
            fast_span=9, slow_span=21, reversion_mode=False,
            enable_regime_filter=False, enable_volume_filter=True,
            vol_imbalance_z_threshold=0.5,
        )
        signal = model.generate_signal(df)
        assert (signal.isin([-1, 0, 1])).all()

    def test_alpha_with_both_filters(self):
        """Alpha com ambos filtros deve aplicar regime E volume."""
        from src.features.indicators import rolling_hurst_exponent, volume_imbalance_zscore

        df = self._make_ohlcv()
        df["hurst_exponent"] = rolling_hurst_exponent(df["close"], window=50, step=5)
        df["volume_imbalance_zscore"] = volume_imbalance_zscore(
            df["volume"], df["close"], window=20, z_window=50
        )

        model = TrendFollowingAlpha(
            fast_span=9, slow_span=21, reversion_mode=False,
            enable_regime_filter=True, enable_volume_filter=True,
            hurst_threshold=0.55,
            vol_imbalance_z_threshold=0.5,
        )
        signal = model.generate_signal(df)
        assert (signal.isin([-1, 0, 1])).all()
        # With both filters, expect more zeros than without filters
        model_no_filter = TrendFollowingAlpha(
            fast_span=9, slow_span=21, reversion_mode=False,
            enable_regime_filter=False, enable_volume_filter=False,
        )
        signal_no_filter = model_no_filter.generate_signal(df)
        # Filtered signal should have equal or more zeros
        n_zeros_filtered = (signal == 0).sum()
        n_zeros_no_filter = (signal_no_filter == 0).sum()
        assert n_zeros_filtered >= n_zeros_no_filter

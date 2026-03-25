"""
Testes para o filtro CUSUM (cusum_filter.py).

Testa detecção de eventos com threshold fixo e adaptativo usando séries
sintéticas com saltos conhecidos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.cusum_filter import adaptive_cusum_events, cusum_events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def flat_series():
    """Série constante (sem eventos esperados)."""
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.Series(100.0, index=dates, name="close")


@pytest.fixture
def series_with_jumps():
    """Série com saltos conhecidos nos índices ~50 e ~150."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    prices = np.full(n, 100.0)

    # Ruído pequeno
    prices += np.cumsum(np.random.randn(n) * 0.01)

    # Saltos grandes
    prices[50] += 5.0   # salto positivo
    prices[51:] += 5.0
    prices[150] -= 5.0  # salto negativo
    prices[151:] -= 5.0

    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def trending_series():
    """Série com tendência e volatilidade crescente."""
    np.random.seed(77)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    trend = np.linspace(100, 120, n)
    noise = np.cumsum(np.random.randn(n) * 0.2)
    return pd.Series(trend + noise, index=dates, name="close")


# ---------------------------------------------------------------------------
# Testes — CUSUM com threshold fixo
# ---------------------------------------------------------------------------
class TestCusumEvents:
    """Testes para CUSUM com threshold fixo."""

    def test_detects_jumps(self, series_with_jumps):
        """Deve detectar eventos nos saltos conhecidos."""
        events = cusum_events(series_with_jumps, threshold=1.0)
        assert len(events) >= 2, f"Esperava ≥2 eventos, obteve {len(events)}"

    def test_flat_series_no_events(self, flat_series):
        """Série constante com threshold alto deve gerar zero eventos."""
        events = cusum_events(flat_series, threshold=100.0)
        assert len(events) == 0

    def test_lower_threshold_more_events(self, series_with_jumps):
        """Threshold menor deve gerar mais (ou igual) eventos."""
        events_high = cusum_events(series_with_jumps, threshold=2.0)
        events_low = cusum_events(series_with_jumps, threshold=0.5)
        assert len(events_low) >= len(events_high)

    def test_returns_datetimeindex(self, series_with_jumps):
        """Resultado deve ser um DatetimeIndex."""
        events = cusum_events(series_with_jumps, threshold=1.0)
        assert isinstance(events, pd.DatetimeIndex)

    def test_events_are_subset_of_index(self, series_with_jumps):
        """Eventos devem ser subset do índice da série diferenciada."""
        events = cusum_events(series_with_jumps, threshold=1.0)
        diff_index = series_with_jumps.diff().dropna().index
        for ts in events:
            assert ts in diff_index


# ---------------------------------------------------------------------------
# Testes — CUSUM adaptativo
# ---------------------------------------------------------------------------
class TestAdaptiveCusumEvents:
    """Testes para CUSUM com threshold adaptativo (EWMA)."""

    def test_generates_events(self, trending_series):
        """Deve gerar eventos em série com tendência."""
        events = adaptive_cusum_events(trending_series, ewm_span=30, threshold_multiplier=1.0)
        assert len(events) > 0

    def test_returns_datetimeindex(self, trending_series):
        """Resultado deve ser DatetimeIndex."""
        events = adaptive_cusum_events(trending_series, ewm_span=30, threshold_multiplier=1.0)
        assert isinstance(events, pd.DatetimeIndex)

    def test_higher_multiplier_fewer_events(self, trending_series):
        """Multiplicador maior → menos eventos."""
        events_low = adaptive_cusum_events(
            trending_series, ewm_span=30, threshold_multiplier=0.5
        )
        events_high = adaptive_cusum_events(
            trending_series, ewm_span=30, threshold_multiplier=3.0
        )
        assert len(events_high) <= len(events_low)

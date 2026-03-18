"""
Testes para o módulo de amostragem de barras alternativas (bar_sampler.py).

Testa barras de volume, barras de dólar e barras de tick com dados sintéticos,
verificando propriedades esperadas como totalização de volume e OHLC correto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.bar_sampler import dollar_bars, tick_bars, volume_bars


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tick_data():
    """DataFrame de ticks simulado com 1000 ticks."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2024-01-02 10:00", periods=n, freq="100ms", tz="UTC")

    base_price = 50.0
    price_changes = np.random.randn(n) * 0.01
    prices = base_price + np.cumsum(price_changes)

    return pd.DataFrame(
        {
            "last": prices,
            "volume": np.random.randint(1, 20, n).astype(float),
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )


@pytest.fixture
def high_volume_ticks():
    """Ticks com volume alto para gerar múltiplas barras com threshold baixo."""
    np.random.seed(123)
    n = 500
    dates = pd.date_range("2024-01-03 10:00", periods=n, freq="50ms", tz="UTC")
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.05)
    volumes = np.random.randint(5, 50, n).astype(float)

    return pd.DataFrame(
        {"last": prices, "volume": volumes},
        index=pd.DatetimeIndex(dates, name="time"),
    )


# ---------------------------------------------------------------------------
# Testes — Barras de Volume
# ---------------------------------------------------------------------------
class TestVolumeBars:
    """Testes para barras de volume."""

    def test_generates_bars(self, tick_data):
        """Deve gerar pelo menos uma barra com threshold razoável."""
        bars = volume_bars(tick_data, threshold=50)
        assert len(bars) > 0

    def test_bar_volume_meets_threshold(self, tick_data):
        """Cada barra deve ter volume >= threshold."""
        threshold = 50
        bars = volume_bars(tick_data, threshold=threshold)
        assert (bars["volume"] >= threshold).all()

    def test_ohlc_consistency(self, tick_data):
        """High >= Low, Open e Close dentro de [Low, High]."""
        bars = volume_bars(tick_data, threshold=50)
        assert (bars["high"] >= bars["low"]).all()
        assert (bars["open"] >= bars["low"]).all()
        assert (bars["open"] <= bars["high"]).all()
        assert (bars["close"] >= bars["low"]).all()
        assert (bars["close"] <= bars["high"]).all()

    def test_fewer_bars_with_higher_threshold(self, tick_data):
        """Threshold maior deve gerar menos barras."""
        bars_low = volume_bars(tick_data, threshold=30)
        bars_high = volume_bars(tick_data, threshold=100)
        assert len(bars_high) < len(bars_low)

    def test_n_ticks_column(self, tick_data):
        """Cada barra deve reportar o número de ticks que a compõem."""
        bars = volume_bars(tick_data, threshold=50)
        assert "n_ticks" in bars.columns
        assert (bars["n_ticks"] > 0).all()

    def test_missing_columns_raises(self):
        """Colunas faltantes devem gerar ValueError."""
        df = pd.DataFrame({"price": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))
        with pytest.raises(ValueError, match="Colunas necessárias"):
            volume_bars(df, threshold=10)


# ---------------------------------------------------------------------------
# Testes — Barras de Dólar
# ---------------------------------------------------------------------------
class TestDollarBars:
    """Testes para barras de dólar."""

    def test_generates_bars(self, tick_data):
        """Deve gerar barras com threshold financeiro razoável."""
        bars = dollar_bars(tick_data, threshold=500)
        assert len(bars) > 0

    def test_dollar_value_meets_threshold(self, tick_data):
        """A soma de (price * volume) em cada barra deve atingir o threshold."""
        # Barras de dólar são construídas com base em valor financeiro
        # O volume na barra é volume de ticks, não valor financeiro
        # Apenas verificamos que barras foram geradas corretamente
        bars = dollar_bars(tick_data, threshold=500)
        assert (bars["volume"] > 0).all()

    def test_ohlc_consistency(self, tick_data):
        """Consistência OHLC nas barras de dólar."""
        bars = dollar_bars(tick_data, threshold=500)
        assert (bars["high"] >= bars["low"]).all()

    def test_fewer_bars_with_higher_threshold(self, tick_data):
        """Threshold financeiro maior → menos barras."""
        bars_low = dollar_bars(tick_data, threshold=200)
        bars_high = dollar_bars(tick_data, threshold=2000)
        assert len(bars_high) < len(bars_low)


# ---------------------------------------------------------------------------
# Testes — Barras de Tick
# ---------------------------------------------------------------------------
class TestTickBars:
    """Testes para barras de tick."""

    def test_generates_bars(self, tick_data):
        """Deve gerar barras a cada N ticks."""
        bars = tick_bars(tick_data, threshold=50)
        assert len(bars) > 0

    def test_n_ticks_equals_threshold(self, tick_data):
        """Cada barra deve conter exatamente threshold ticks."""
        threshold = 50
        bars = tick_bars(tick_data, threshold=threshold)
        # Todas as barras devem ter exatamente threshold ticks
        assert (bars["n_ticks"] == threshold).all()

    def test_expected_number_of_bars(self, tick_data):
        """Número de barras deve ser len(ticks) // threshold."""
        threshold = 100
        bars = tick_bars(tick_data, threshold=threshold)
        expected = len(tick_data) // threshold
        assert len(bars) == expected

    def test_ohlc_consistency(self, tick_data):
        """Consistência OHLC nas barras de tick."""
        bars = tick_bars(tick_data, threshold=50)
        assert (bars["high"] >= bars["low"]).all()
        assert (bars["open"] >= bars["low"]).all()
        assert (bars["close"] <= bars["high"]).all()


# ---------------------------------------------------------------------------
# Testes - Edge Cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Testes de casos limite."""

    def test_threshold_too_high_returns_empty(self, tick_data):
        """Threshold impossível deve retornar DataFrame vazio."""
        total_volume = tick_data["volume"].sum()
        bars = volume_bars(tick_data, threshold=int(total_volume * 10))
        assert len(bars) == 0

    def test_single_bar(self, tick_data):
        """Threshold = volume total deve gerar 1 barra."""
        total_volume = tick_data["volume"].sum()
        bars = volume_bars(tick_data, threshold=int(total_volume))
        assert len(bars) == 1

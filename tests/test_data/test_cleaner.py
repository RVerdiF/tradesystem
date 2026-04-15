"""Testes para o módulo de limpeza de dados (cleaner.py).

Testa remoção de spikes, preenchimento de lacunas e validação OHLC
com dados sintéticos controlados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import (
    clean_ohlc,
    clean_ticks,
    fill_gaps,
    remove_spikes,
    remove_tick_spikes,
    validate_ohlc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_ohlc_df():
    """DataFrame OHLC limpo (sem problemas)."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.05,
            "high": close + abs(np.random.randn(n) * 0.2),
            "low": close - abs(np.random.randn(n) * 0.2),
            "close": close,
            "volume": np.random.randint(100, 1000, n),
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )


@pytest.fixture
def ohlc_with_spikes(clean_ohlc_df):
    """DataFrame OHLC com spikes conhecidos."""
    df = clean_ohlc_df.copy()
    # Injeta spike no índice 30 e 70
    df.iloc[30, df.columns.get_loc("close")] = 999.0  # spike grande
    df.iloc[70, df.columns.get_loc("close")] = 0.01   # spike baixo
    return df


@pytest.fixture
def tick_df():
    """DataFrame de ticks simulado."""
    n = 500
    dates = pd.date_range("2024-01-02 10:00", periods=n, freq="100ms", tz="UTC")
    last = 50.0 + np.cumsum(np.random.randn(n) * 0.01)
    return pd.DataFrame(
        {
            "last": last,
            "volume": np.random.randint(1, 10, n),
            "bid": last - 0.01,
            "ask": last + 0.01,
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )


# ---------------------------------------------------------------------------
# Testes — Remoção de Spikes
# ---------------------------------------------------------------------------
class TestRemoveSpikes:
    """Testes de remove_spikes (Z-score rolling)."""

    def test_removes_known_spikes(self, ohlc_with_spikes):
        """Spikes injetados devem ser removidos."""
        cleaned = remove_spikes(ohlc_with_spikes, z_threshold=3.0, window=20)
        assert len(cleaned) < len(ohlc_with_spikes)

    def test_preserves_clean_data(self, clean_ohlc_df):
        """Dados limpos não devem sofrer remoção."""
        cleaned = remove_spikes(clean_ohlc_df, z_threshold=5.0, window=20)
        assert len(cleaned) == len(clean_ohlc_df)

    def test_missing_column_returns_unchanged(self, clean_ohlc_df):
        """Coluna inexistente deve retornar dados inalterados."""
        cleaned = remove_spikes(clean_ohlc_df, price_col="nonexistent")
        assert len(cleaned) == len(clean_ohlc_df)


class TestRemoveTickSpikes:
    """Testes de remove_tick_spikes (retorno máximo)."""

    def test_removes_impossible_returns(self, tick_df):
        """Ticks com retornos impossíveis devem ser removidos."""
        df = tick_df.copy()
        # Injeta salto de 50%
        df.iloc[100, df.columns.get_loc("last")] = df.iloc[99]["last"] * 1.5
        cleaned = remove_tick_spikes(df, max_return_pct=1.0)
        assert len(cleaned) < len(df)

    def test_preserves_normal_ticks(self, tick_df):
        """Ticks com retornos normais devem ser preservados."""
        cleaned = remove_tick_spikes(tick_df, max_return_pct=5.0)
        assert len(cleaned) == len(tick_df)


# ---------------------------------------------------------------------------
# Testes — Preenchimento de Lacunas
# ---------------------------------------------------------------------------
class TestFillGaps:
    """Testes de fill_gaps."""

    def test_fills_missing_periods(self):
        """Lacunas devem ser preenchidas com forward fill."""
        dates = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05",
                                 "2024-01-01 10:15", "2024-01-01 10:20"])
        df = pd.DataFrame({"close": [100, 101, 103, 104]}, index=dates)
        df.index = df.index.tz_localize("UTC")

        filled = fill_gaps(df, method="ffill", freq="5min", max_gap=10)

        # 10:10 estava faltando — deve ser preenchido
        assert len(filled) == 5  # 10:00, 10:05, 10:10, 10:15, 10:20

    def test_respects_max_gap(self):
        """Lacunas maiores que max_gap devem ficar como NaN."""
        dates = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"])
        df = pd.DataFrame({"close": [100, 110]}, index=dates)
        df.index = df.index.tz_localize("UTC")

        filled = fill_gaps(df, method="ffill", freq="5min", max_gap=3)

        # Lacuna muito grande — muitos NaNs restantes
        assert filled["close"].isna().sum() > 0


# ---------------------------------------------------------------------------
# Testes — Validação OHLC
# ---------------------------------------------------------------------------
class TestValidateOHLC:
    """Testes de validate_ohlc."""

    def test_valid_data_passes(self, clean_ohlc_df):
        """Dados corretos não devem gerar problemas."""
        # Primeiro garante que os dados estão realmente válidos
        df = clean_ohlc_df.copy()
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

        result = validate_ohlc(df)
        assert len(result) == len(df)

    def test_fixes_high_low_swap(self):
        """High < Low deve ser corrigido com swap."""
        df = pd.DataFrame({
            "open": [100.0],
            "high": [95.0],   # errado — menor que low
            "low": [105.0],   # errado — maior que high
            "close": [100.0],
        })
        fixed = validate_ohlc(df, fix=True)
        assert fixed["high"].iloc[0] == 105.0
        assert fixed["low"].iloc[0] == 95.0

    def test_clips_open_to_range(self):
        """Open fora de [Low, High] deve ser clipado."""
        df = pd.DataFrame({
            "open": [200.0],   # fora do range
            "high": [110.0],
            "low": [90.0],
            "close": [100.0],
        })
        fixed = validate_ohlc(df, fix=True)
        assert 90.0 <= fixed["open"].iloc[0] <= 110.0


# ---------------------------------------------------------------------------
# Testes — Pipeline Completo
# ---------------------------------------------------------------------------
class TestCleanPipelines:
    """Testes dos pipelines clean_ohlc e clean_ticks."""

    def test_clean_ohlc_runs(self, ohlc_with_spikes):
        """Pipeline OHLC completo deve executar sem erros."""
        result = clean_ohlc(ohlc_with_spikes, z_threshold=3.0)
        assert len(result) > 0
        # Pipeline deve detectar e remover os spikes injetados (999.0 e 0.01)
        assert 999.0 not in result["close"].values
        assert 0.01 not in result["close"].values

    def test_clean_ticks_runs(self, tick_df):
        """Pipeline de ticks deve executar sem erros."""
        result = clean_ticks(tick_df, max_return_pct=2.0)
        assert len(result) > 0

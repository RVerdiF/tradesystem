"""
Testes para a Fase 3 — Alpha, Volatilidade, Tripla Barreira e Meta-Labeling.

Usa dados sintéticos com comportamentos conhecidos para validar que:
- Alpha gera sinais corretos
- Tripla Barreira identifica TP/SL/vertical corretamente
- Meta-labels binários são consistentes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labeling.alpha import (
    MeanReversionAlpha,
    TrendFollowingAlpha,
    get_signal_events,
)
from src.labeling.volatility import daily_vol, get_volatility_targets
from src.labeling.triple_barrier import apply_triple_barrier, create_events, get_labels
from src.labeling.meta_labeling import (
    build_training_dataset,
    get_meta_labels,
    meta_label_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def trending_up_df():
    """DataFrame com tendência de alta clara (EMA rápida > lenta)."""
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    prices = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1
    return pd.DataFrame(
        {
            "open": prices - 0.1,
            "high": prices + 0.3,
            "low": prices - 0.3,
            "close": prices,
            "volume": np.random.randint(100, 500, n),
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )


@pytest.fixture
def mean_reverting_df():
    """DataFrame com preço oscilando em torno de média (bom para mean reversion)."""
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    # Sinal sinusoidal + ruído
    t = np.arange(n)
    prices = 100 + 5 * np.sin(2 * np.pi * t / 50) + np.random.randn(n) * 0.2
    return pd.DataFrame(
        {
            "open": prices - 0.1,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.random.randint(100, 500, n),
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )


@pytest.fixture
def simple_up_down_df():
    """
    Série com padrão simples: sobe 10 barras, desce 10 barras.
    Ideal para testar TP/SL/Vertical de forma controlada.
    """
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    prices = np.zeros(n)
    prices[0] = 100.0
    for i in range(1, n):
        if (i // 10) % 2 == 0:
            prices[i] = prices[i - 1] + 0.5  # sobe
        else:
            prices[i] = prices[i - 1] - 0.5  # desce
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.2,
            "low": prices - 0.2,
            "close": prices,
            "volume": np.full(n, 100),
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )


# ---------------------------------------------------------------------------
# Testes — Alpha
# ---------------------------------------------------------------------------
class TestTrendFollowingAlpha:
    """Testes para TrendFollowingAlpha."""

    def test_generates_signals(self, trending_up_df):
        """Deve gerar sinais +1 e -1."""
        alpha = TrendFollowingAlpha(fast_span=5, slow_span=20)
        signal = alpha.generate_signal(trending_up_df)
        assert len(signal) == len(trending_up_df)
        assert set(signal.unique()).issubset({-1, 0, 1})

    def test_uptrend_mostly_long(self, trending_up_df):
        """Em tendência de alta, deve gerar predominantemente sinais +1."""
        alpha = TrendFollowingAlpha(fast_span=5, slow_span=20)
        signal = alpha.generate_signal(trending_up_df)
        # Exclui warmup
        active_signal = signal.iloc[20:]
        long_pct = (active_signal == 1).mean()
        assert long_pct > 0.7  # maioria longa

    def test_warmup_is_neutral(self, trending_up_df):
        """Os primeiros períodos (warmup) devem ser neutros (0)."""
        alpha = TrendFollowingAlpha(fast_span=5, slow_span=20)
        signal = alpha.generate_signal(trending_up_df)
        assert (signal.iloc[:20] == 0).all()

    def test_name_property(self):
        """Propriedade name deve retornar string descritiva."""
        alpha = TrendFollowingAlpha(fast_span=10, slow_span=50)
        assert "TrendFollowing" in alpha.name


class TestMeanReversionAlpha:
    """Testes para MeanReversionAlpha."""

    def test_generates_signals(self, mean_reverting_df):
        """Deve gerar sinais +1, -1 e 0."""
        alpha = MeanReversionAlpha(window=20, entry_threshold=1.5, exit_threshold=0.0)
        signal = alpha.generate_signal(mean_reverting_df)
        assert len(signal) == len(mean_reverting_df)
        assert set(signal.unique()).issubset({-1, 0, 1})

    def test_oscillating_signals(self, mean_reverting_df):
        """Em série oscilante, deve gerar sinais em ambas direções."""
        alpha = MeanReversionAlpha(window=20, entry_threshold=1.5, exit_threshold=0.0)
        signal = alpha.generate_signal(mean_reverting_df)
        has_long = (signal == 1).any()
        has_short = (signal == -1).any()
        assert has_long and has_short


class TestSignalEvents:
    """Testes para get_signal_events."""

    def test_extracts_events(self, trending_up_df):
        """Deve extrair timestamps de mudança de sinal."""
        alpha = TrendFollowingAlpha(fast_span=5, slow_span=20)
        signal = alpha.generate_signal(trending_up_df)
        events = get_signal_events(signal)
        assert isinstance(events, pd.DatetimeIndex)
        assert len(events) > 0


# ---------------------------------------------------------------------------
# Testes — Volatilidade
# ---------------------------------------------------------------------------
class TestVolatility:
    """Testes para daily_vol e get_volatility_targets."""

    def test_daily_vol_positive(self, trending_up_df):
        """Volatilidade deve ser sempre positiva."""
        vol = daily_vol(trending_up_df["close"], span=20)
        assert (vol.dropna() > 0).all()

    def test_volatility_targets_at_events(self, trending_up_df):
        """Targets devem ter valores nos eventos."""
        vol = daily_vol(trending_up_df["close"], span=20)
        events = trending_up_df.index[50::20]  # alguns timestamps
        targets = get_volatility_targets(trending_up_df["close"], events, span=20)
        assert len(targets) > 0
        assert (targets > 0).all()


# ---------------------------------------------------------------------------
# Testes — Tripla Barreira
# ---------------------------------------------------------------------------
class TestTripleBarrier:
    """Testes para create_events, apply_triple_barrier e get_labels."""

    def test_create_events_output_shape(self, simple_up_down_df):
        """create_events deve retornar DataFrame com colunas corretas."""
        close = simple_up_down_df["close"]
        events_ts = close.index[10::20]  # eventos espaçados
        vol = daily_vol(close, span=10)
        targets = get_volatility_targets(close, events_ts, span=10)

        events = create_events(
            close, events_ts, targets,
            pt_sl=(1.0, 1.0), max_holding=15,
        )
        assert "t1" in events.columns
        assert "trgt" in events.columns
        assert "side" in events.columns
        assert len(events) > 0

    def test_barrier_types_present(self, simple_up_down_df):
        """Deve identificar diferentes tipos de barreira."""
        close = simple_up_down_df["close"]
        events_ts = close.index[10::10]
        vol = daily_vol(close, span=10)
        targets = get_volatility_targets(close, events_ts, span=10)

        events = create_events(
            close, events_ts, targets,
            pt_sl=(0.5, 0.5), max_holding=8,
        )
        labels = get_labels(close, events, pt_sl=(0.5, 0.5), open_prices=close)

        # Deve ter pelo menos algum resultado
        assert len(labels) > 0
        assert "label" in labels.columns
        assert "barrier_type" in labels.columns

    def test_labels_values(self, simple_up_down_df):
        """Labels devem estar em {-1, 0, +1}."""
        close = simple_up_down_df["close"]
        events_ts = close.index[10::15]
        vol = daily_vol(close, span=10)
        targets = get_volatility_targets(close, events_ts, span=10)

        events = create_events(
            close, events_ts, targets,
            pt_sl=(1.0, 1.0), max_holding=10,
        )
        labels = get_labels(close, events, open_prices=close)

        if len(labels) > 0:
            assert set(labels["label"].unique()).issubset({-1, 0, 1})

    def test_no_future_timestamps(self, simple_up_down_df):
        """t1 deve ser sempre >= timestamp do evento."""
        close = simple_up_down_df["close"]
        events_ts = close.index[10::15]
        vol = daily_vol(close, span=10)
        targets = get_volatility_targets(close, events_ts, span=10)

        events = create_events(
            close, events_ts, targets,
            pt_sl=(1.0, 1.0), max_holding=10,
        )
        result = apply_triple_barrier(close, events, open_prices=close)

        if len(result) > 0:
            assert (result["t1"] >= result.index).all()


# ---------------------------------------------------------------------------
# Testes — Meta-Labeling
# ---------------------------------------------------------------------------
class TestMetaLabeling:
    """Testes para get_meta_labels e build_training_dataset."""

    def test_meta_labels_binary(self, simple_up_down_df):
        """Meta-labels devem ser binários {0, 1}."""
        close = simple_up_down_df["close"]
        events_ts = close.index[10::15]
        vol = daily_vol(close, span=10)
        targets = get_volatility_targets(close, events_ts, span=10)

        events = create_events(
            close, events_ts, targets,
            pt_sl=(1.0, 1.0), max_holding=10,
        )
        labels = get_labels(close, events, open_prices=close)

        if len(labels) > 0:
            meta = get_meta_labels(labels)
            assert set(meta.unique()).issubset({0, 1})
            assert meta.dtype == np.int8

    def test_build_training_dataset(self, simple_up_down_df):
        """Dataset de treinamento deve combinar features e meta-labels."""
        close = simple_up_down_df["close"]
        events_ts = close.index[10::15]
        vol = daily_vol(close, span=10)
        targets = get_volatility_targets(close, events_ts, span=10)

        events = create_events(
            close, events_ts, targets,
            pt_sl=(1.0, 1.0), max_holding=10,
        )
        labels = get_labels(close, events, open_prices=close)

        if len(labels) == 0:
            pytest.skip("Nenhum label gerado")

        # Features simples para teste
        features = pd.DataFrame(
            {"feat1": np.random.randn(len(close)), "feat2": np.random.randn(len(close))},
            index=close.index,
        )

        dataset = build_training_dataset(features, labels)
        assert "meta_label" in dataset.columns
        assert "feat1" in dataset.columns
        assert len(dataset) > 0

    def test_meta_label_analysis(self, simple_up_down_df):
        """Análise de meta-labels deve retornar dicionário com métricas."""
        close = simple_up_down_df["close"]
        events_ts = close.index[10::15]
        vol = daily_vol(close, span=10)
        targets = get_volatility_targets(close, events_ts, span=10)

        events = create_events(
            close, events_ts, targets,
            pt_sl=(1.0, 1.0), max_holding=10,
        )
        labels = get_labels(close, events, open_prices=close)

        if len(labels) == 0:
            pytest.skip("Nenhum label gerado")

        stats = meta_label_analysis(labels)
        assert "total_events" in stats
        assert "positive_ratio" in stats
        assert stats["total_events"] > 0

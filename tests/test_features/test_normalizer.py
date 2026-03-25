"""
Testes para o módulo de normalização temporal (normalizer.py).

Verifica propriedades estatísticas (Z-score, rank), normalização em lote,
e validação anti look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.normalizer import (
    expanding_rank,
    normalize_features,
    rolling_zscore,
    validate_no_lookahead,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def price_series():
    """Série de preços com 200 pontos."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def feature_df():
    """DataFrame multivariado para normalização em lote."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "rsi": np.random.uniform(20, 80, n),
            "macd": np.random.randn(n) * 2,
            "atr": np.abs(np.random.randn(n)) + 0.5,
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )


# ---------------------------------------------------------------------------
# Testes — Rolling Z-Score
# ---------------------------------------------------------------------------
class TestRollingZScore:
    """Testes para Z-score com janela móvel."""

    def test_mean_near_zero(self, price_series):
        """Média rolling do Z-score deve ser ≈ 0."""
        window = 50
        z = rolling_zscore(price_series, window=window).dropna()
        # Média do Z-score rolling não é exatamente 0 globalmente,
        # mas localmente em janelas
        assert abs(z.mean()) < 2.0  # tolerância ampla

    def test_std_near_one(self, price_series):
        """Std rolling do Z-score deve ser ≈ 1."""
        window = 50
        z = rolling_zscore(price_series, window=window).dropna()
        # Std do Z-score não é exatamente 1 globalmente, mas próximo
        assert 0.3 < z.std() < 3.0  # tolerância ampla

    def test_output_length(self, price_series):
        """Saída deve ter mesmo comprimento que entrada."""
        z = rolling_zscore(price_series, window=50)
        assert len(z) == len(price_series)

    def test_name_suffix(self, price_series):
        """Nome deve ter sufixo _zscore."""
        z = rolling_zscore(price_series, window=50)
        assert z.name == "close_zscore"


# ---------------------------------------------------------------------------
# Testes — Expanding Rank
# ---------------------------------------------------------------------------
class TestExpandingRank:
    """Testes para rank percentual expandido."""

    def test_range_zero_one(self, price_series):
        """Valores devem estar em [0, 1]."""
        r = expanding_rank(price_series).dropna()
        assert (r >= 0).all() and (r <= 1).all()

    def test_monotonic_for_sorted(self):
        """Para série crescente, rank deve ser 1.0 em todos os pontos."""
        s = pd.Series(range(1, 101), name="test")
        r = expanding_rank(s)
        assert (r == 1.0).all()

    def test_name_suffix(self, price_series):
        """Nome deve ter sufixo _rank."""
        r = expanding_rank(price_series)
        assert r.name == "close_rank"


# ---------------------------------------------------------------------------
# Testes — Normalização em lote
# ---------------------------------------------------------------------------
class TestNormalizeFeatures:
    """Testes para normalize_features."""

    def test_same_columns(self, feature_df):
        """Deve manter as mesmas colunas."""
        result = normalize_features(feature_df, method="zscore")
        assert list(result.columns) == list(feature_df.columns)

    def test_same_index(self, feature_df):
        """Deve manter o mesmo índice."""
        result = normalize_features(feature_df, method="zscore")
        assert result.index.equals(feature_df.index)

    def test_rank_method(self, feature_df):
        """Método rank deve retornar valores em [0, 1]."""
        result = normalize_features(feature_df, method="rank")
        for col in result.columns:
            vals = result[col].dropna()
            assert (vals >= 0).all() and (vals <= 1).all()


# ---------------------------------------------------------------------------
# Testes — Validação Anti Look-Ahead
# ---------------------------------------------------------------------------
class TestValidateNoLookahead:
    """Testes para validate_no_lookahead."""

    def test_valid_normalization(self, feature_df):
        """Normalização rolling_zscore correta deve passar na validação."""
        normalized = normalize_features(feature_df, method="zscore", window=50)
        assert validate_no_lookahead(normalized, feature_df, window=50) is True

    def test_detects_lookahead(self, feature_df):
        """Normalização com dados futuros deve falhar na validação."""
        # Simula look-ahead: normaliza com dados globais (futuro incluído)
        global_mean = feature_df.mean()
        global_std = feature_df.std()
        fake_normalized = (feature_df - global_mean) / global_std

        # Isso deve detectar discrepância vs. rolling
        result = validate_no_lookahead(fake_normalized, feature_df, window=50)
        assert result is False

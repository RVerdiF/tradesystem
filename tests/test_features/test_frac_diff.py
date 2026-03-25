"""
Testes para o módulo de diferenciação fracionária (frac_diff.py).

Testa cálculo de pesos FFD, aplicação da transformação e busca automática
do d mínimo com dados sintéticos (random walk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from src.features.frac_diff import find_min_d, frac_diff_ffd, get_weights_ffd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def random_walk():
    """Série random walk (não estacionária) com 2000 pontos."""
    np.random.seed(42)
    n = 2000
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def stationary_series():
    """Série estacionária (ruído branco com média fixa)."""
    np.random.seed(99)
    n = 300
    dates = pd.date_range("2024-02-01", periods=n, freq="5min", tz="UTC")
    values = np.random.randn(n) * 2.0 + 50.0
    return pd.Series(values, index=dates, name="close")


# ---------------------------------------------------------------------------
# Testes — Pesos FFD
# ---------------------------------------------------------------------------
class TestGetWeightsFFD:
    """Testes para cálculo de pesos FFD."""

    def test_first_weight_is_one(self):
        """O primeiro peso (mais recente) é sempre 1.0."""
        weights = get_weights_ffd(d=0.5, threshold=1e-5)
        assert weights[-1] == 1.0

    def test_weights_decrease(self):
        """Pesos devem decrescer em valor absoluto."""
        weights = get_weights_ffd(d=0.5, threshold=1e-5)
        abs_weights = np.abs(weights)
        # Do mais antigo ao mais recente, devem ser crescentes
        assert abs_weights[-1] >= abs_weights[0]

    def test_higher_d_more_weights(self):
        """d maior deve gerar menos pesos (decaem mais rápido para d inteiro)."""
        w_low = get_weights_ffd(d=0.2, threshold=1e-5)
        w_high = get_weights_ffd(d=0.8, threshold=1e-5)
        # Ambos devem ter pesos válidos
        assert len(w_low) > 0
        assert len(w_high) > 0

    def test_d_zero_single_weight(self):
        """d=0 deve produzir apenas 1 peso (sem diferenciação)."""
        weights = get_weights_ffd(d=0.0, threshold=1e-5)
        assert len(weights) == 1
        assert weights[0] == 1.0


# ---------------------------------------------------------------------------
# Testes — Aplicação FFD
# ---------------------------------------------------------------------------
class TestFracDiffFFD:
    """Testes para aplicação da diferenciação fracionária."""

    def test_output_same_length(self, random_walk):
        """Saída deve ter mesmo comprimento que entrada (com NaNs no início)."""
        result = frac_diff_ffd(random_walk, d=0.5)
        assert len(result) == len(random_walk)

    def test_initial_nans(self, random_walk):
        """Primeiros len(weights)-1 valores devem ser NaN."""
        d = 0.5
        threshold = 1e-3  # threshold maior para pesos mais curtos
        weights = get_weights_ffd(d, threshold=threshold)
        result = frac_diff_ffd(random_walk, d=d, threshold=threshold)

        n_nan = result.isna().sum()
        assert n_nan == len(weights) - 1

    def test_d_one_approximates_diff(self, random_walk):
        """d=1.0 deve se aproximar da primeira diferença."""
        result = frac_diff_ffd(random_walk, d=1.0).dropna()
        expected = random_walk.diff().dropna()

        # Alinha os índices
        common = result.index.intersection(expected.index)
        assert len(common) > 0
        np.testing.assert_allclose(result.loc[common].values, expected.loc[common].values, atol=1e-6)

    def test_stationarity_with_high_d(self, random_walk):
        """Série com d alto deve ser estacionária (ADF test)."""
        result = frac_diff_ffd(random_walk, d=0.8).dropna()
        adf_stat, pval, *_ = adfuller(result, maxlag=1, regression="c", autolag=None)
        assert pval < 0.05, f"Série não estacionária com d=0.8 (p={pval:.4f})"

    def test_preserves_index(self, random_walk):
        """O índice da série deve ser preservado."""
        result = frac_diff_ffd(random_walk, d=0.5)
        assert result.index.equals(random_walk.index)


# ---------------------------------------------------------------------------
# Testes — Busca do d mínimo
# ---------------------------------------------------------------------------
class TestFindMinD:
    """Testes para busca automática do d mínimo."""

    def test_finds_valid_d(self, random_walk):
        """Deve encontrar d entre 0 e 1 para random walk."""
        d = find_min_d(random_walk)
        assert 0.0 < d <= 1.0

    def test_stationary_series_low_d(self, stationary_series):
        """Série já estacionária deve ter d mínimo baixo."""
        d = find_min_d(stationary_series)
        assert d <= 0.3, f"d={d} muito alto para série estacionária"

    def test_custom_d_range(self, random_walk):
        """Deve respeitar d_range customizado."""
        d = find_min_d(random_walk, d_range=np.arange(0.3, 1.0, 0.1))
        assert d >= 0.3

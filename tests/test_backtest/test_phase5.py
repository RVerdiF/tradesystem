"""
Testes para a Fase 5 — Backtesting.

Valida CPCV, modelagem de custos, métricas de performance e atribuição.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.cpcv import CombinatorialPurgedCV
from src.backtest.cost_model import BrazilianCostModel, SlippageModel, total_cost
from src.backtest.metrics import (
    sharpe_ratio,
    deflated_sharpe,
    max_drawdown,
    calmar_ratio,
    probability_of_ruin,
    performance_report,
)
from src.backtest.attribution import (
    attribution_analysis,
    trade_level_attribution,
    attribution_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def positive_returns():
    """Série de retornos positivos consistentes."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.001, 0.01, 252), name="returns")


@pytest.fixture
def negative_returns():
    """Série de retornos negativos (estratégia perdedora)."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(-0.002, 0.01, 252), name="returns")


@pytest.fixture
def synthetic_events_bt():
    """Eventos sintéticos para CPCV."""
    t0 = pd.date_range("2024-01-01", periods=60, freq="5min")
    t1 = t0 + pd.Timedelta(minutes=3)
    return pd.Series(index=t0, data=t1)


# ---------------------------------------------------------------------------
# Testes — CPCV
# ---------------------------------------------------------------------------
class TestCPCV:

    def test_n_paths(self):
        """Número de caminhos deve ser C(n, k)."""
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        assert cpcv.n_paths == 15  # C(6,2) = 15

    def test_split_no_overlap(self, synthetic_events_bt):
        """Treino e teste não devem ter índices em comum."""
        X = pd.DataFrame(
            np.random.randn(len(synthetic_events_bt), 3),
            index=synthetic_events_bt.index,
        )
        # Sem purga para simplificar o teste de não-sobreposição
        cpcv = CombinatorialPurgedCV(
            n_groups=4, n_test_groups=2,
            samples_info=None,  # sem purga
            pct_embargo=0.0,
        )
        splits = cpcv.split(X)

        assert len(splits) == 6  # C(4,2) = 6

        for train_idx, test_idx in splits:
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            assert len(train_idx) > 0
            assert len(test_idx) > 0


# ---------------------------------------------------------------------------
# Testes — Modelo de Custos
# ---------------------------------------------------------------------------
class TestCostModel:

    def test_round_trip_cost(self):
        """Custo de round-trip deve incluir todas as taxas."""
        model = BrazilianCostModel(
            brokerage=0.50,
            emoluments_pct=0.00005,
            settlement_pct=0.0000275,
            iss_pct=0.05,
            symbol="PETR4"
        )
        cost = model.trade_cost(price=100.0, quantity=10, n_operations=2)

        # Corretagem: 0.50 * 2 = 1.00
        # Emolumentos: 1000 * 0.00005 * 2 = 0.10
        # Liquidação: 1000 * 0.0000275 * 2 = 0.055
        # ISS: 1.00 * 0.05 = 0.05
        expected = 1.00 + 0.10 + 0.055 + 0.05
        assert np.isclose(cost, expected, atol=1e-6)

    def test_zero_brokerage(self):
        """Corretagem zero (corretoras modernas) deve funcionar."""
        model = BrazilianCostModel(brokerage=0.0, symbol="PETR4")
        cost = model.trade_cost(price=50.0, quantity=100)
        assert cost >= 0  # Ainda tem emolumentos e liquidação

    def test_slippage_increases_with_participation(self):
        """Slippage deve aumentar com participação no volume."""
        slip = SlippageModel(base_slippage_bps=1.0, symbol="PETR4")
        small = slip.estimate(100.0, 10, avg_volume=1_000_000)
        large = slip.estimate(100.0, 10_000, avg_volume=1_000_000)
        assert large > small


# ---------------------------------------------------------------------------
# Testes — Métricas
# ---------------------------------------------------------------------------
class TestMetrics:

    def test_sharpe_positive(self, positive_returns):
        """Retornos positivos devem ter Sharpe > 0."""
        sr = sharpe_ratio(positive_returns)
        assert sr > 0

    def test_sharpe_constant_returns(self):
        """Retornos constantes: std ≈ 0 → SR deve retornar 0 (guard)."""
        returns = pd.Series([0.01] * 100)
        sr = sharpe_ratio(returns)
        assert sr == 0.0

    def test_max_drawdown_negative(self, positive_returns):
        """MDD deve ser <= 0."""
        mdd = max_drawdown(positive_returns)
        assert mdd <= 0

    def test_deflated_sharpe_penalizes_trials(self, positive_returns):
        """Mais trials devem resultar em p-value maior (menos significativo)."""
        sr = sharpe_ratio(positive_returns)
        p1 = deflated_sharpe(sr, n_trials=1, n_obs=len(positive_returns))
        p10 = deflated_sharpe(sr, n_trials=10, n_obs=len(positive_returns))
        p100 = deflated_sharpe(sr, n_trials=100, n_obs=len(positive_returns))
        # Mais trials → pior p-value
        assert p10 >= p1
        assert p100 >= p10

    def test_calmar_ratio(self, positive_returns):
        """Calmar ratio deve ser positivo para retornos positivos."""
        cr = calmar_ratio(positive_returns)
        assert cr > 0

    def test_performance_report(self, positive_returns):
        """Relatório completo deve conter todas as métricas."""
        report = performance_report(positive_returns, n_trials=5)
        assert "sharpe_ratio" in report
        assert "max_drawdown" in report
        assert "calmar_ratio" in report
        assert "deflated_sharpe_pvalue" in report
        assert "prob_ruin_50pct" in report


# ---------------------------------------------------------------------------
# Testes — Atribuição
# ---------------------------------------------------------------------------
class TestAttribution:

    def test_attribution_analysis(self, positive_returns):
        """Deve decompor SR em Alpha e Meta-Label."""
        rng = np.random.default_rng(99)
        alpha_only = pd.Series(rng.normal(0.0005, 0.012, 252))

        result = attribution_analysis(positive_returns, alpha_only)
        assert "sharpe_full_system" in result
        assert "sharpe_alpha_only" in result
        assert "sharpe_lift_meta" in result

    def test_trade_level_attribution(self):
        """Deve calcular contribuição por trade."""
        trades = pd.DataFrame({
            "ret": [0.02, -0.01, 0.015, -0.005],
            "side": [1, -1, 1, -1],
            "meta_label": [1, 1, 0, 1],
            "bet_size": [0.5, 0.3, 0.0, 0.4],
            "cost": [0.001, 0.001, 0.0, 0.001],
        })
        result = trade_level_attribution(trades)
        assert "alpha_contribution" in result.columns
        assert "net_return" in result.columns
        assert "sized_return" in result.columns

    def test_attribution_summary(self):
        """Sumário deve conter métricas agregadas."""
        trades = pd.DataFrame({
            "ret": [0.02, -0.01, 0.015],
            "side": [1, -1, 1],
            "meta_label": [1, 1, 0],
            "bet_size": [0.5, 0.3, 0.0],
        })
        attr = trade_level_attribution(trades)
        summary = attribution_summary(attr)
        assert "total_trades" in summary
        assert "alpha_win_rate" in summary

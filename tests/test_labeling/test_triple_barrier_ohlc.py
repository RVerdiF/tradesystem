"""Testes para a semântica v2 da Tripla Barreira (avaliação intrabar High/Low).

Cobre os quatro cenários distintos que a versão close-only não conseguia
detectar corretamente:
  1. PT tocado pela Máxima, fechamento reverte.
  2. SL tocado pela Mínima, fechamento recupera.
  3. Ambiguidade same-bar (PT e SL cruzados) → SL vence.
  4. BE ativado pela Máxima e SL atingido na mesma barra.
"""
import numpy as np
import pandas as pd
import pytest

from src.labeling.triple_barrier import (
    apply_triple_barrier,
    create_events,
    get_labels,
)


def _build_ohlc(bars: list[dict]) -> dict[str, pd.Series]:
    """Helper: converte lista de dicts OHLC em Series indexadas por timestamp."""
    idx = pd.date_range("2026-01-01 10:00", periods=len(bars), freq="1min")
    df = pd.DataFrame(bars, index=idx)
    return {
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["close"],
    }


def test_pt_hit_by_high_close_reverts():
    """Cenário A: Long. Entry @ bar1 open = 100. PT = +2%, SL = -2%.
    Bar 3: high 102.5 (cruza PT), close 100.5 (reverte).
    Semântica v1 (buggy): não detecta — continua até vertical.
    Semântica v2: detecta PT na bar 3, retorna ret = +0.02.
    """
    bars = [
        {"open": 100, "high": 100.1, "low": 99.9, "close": 100},
        {"open": 100, "high": 100.5, "low": 99.9, "close": 100.2},  # bar1 (entry)
        {"open": 100.2, "high": 100.8, "low": 100.0, "close": 100.3},
        {"open": 100.3, "high": 102.5, "low": 100.2, "close": 100.5},  # PT crossed by high
        {"open": 100.5, "high": 100.7, "low": 100.3, "close": 100.4},
    ] + [{"open": 100.4, "high": 100.5, "low": 100.3, "close": 100.4}] * 5

    ohlc = _build_ohlc(bars)
    close = ohlc["close"]

    events = pd.DataFrame({
        "t1": [close.index[9]],
        "trgt": [0.02],
        "side": [1],
    }, index=[close.index[0]])

    result = apply_triple_barrier(
        close, events, pt_sl=(1.0, 1.0),
        open_prices=ohlc["open"],
        high_prices=ohlc["high"],
        low_prices=ohlc["low"],
    )

    assert len(result) == 1
    assert result.iloc[0]["barrier_type"] == 0  # pt
    assert result.iloc[0]["ret"] == pytest.approx(0.02, abs=1e-9)


def test_sl_hit_by_low_close_recovers():
    """Cenário B: Long. Bar 3: low 97.5 (cruza SL), close 100 (recupera).
    Semântica v2: SL na bar 3, ret = -0.02.
    """
    bars = [
        {"open": 100, "high": 100.1, "low": 99.9, "close": 100},
        {"open": 100, "high": 100.3, "low": 99.8, "close": 100.1},  # bar1 (entry)
        {"open": 100.1, "high": 100.2, "low": 99.7, "close": 99.9},
        {"open": 99.9, "high": 100.0, "low": 97.5, "close": 100.0},  # SL crossed by low
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0},
    ] + [{"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0}] * 5

    ohlc = _build_ohlc(bars)
    close = ohlc["close"]

    events = pd.DataFrame({
        "t1": [close.index[9]],
        "trgt": [0.02],
        "side": [1],
    }, index=[close.index[0]])

    result = apply_triple_barrier(
        close, events, pt_sl=(1.0, 1.0),
        open_prices=ohlc["open"],
        high_prices=ohlc["high"],
        low_prices=ohlc["low"],
    )

    assert len(result) == 1
    assert result.iloc[0]["barrier_type"] == 1  # sl
    assert result.iloc[0]["ret"] == pytest.approx(-0.02, abs=1e-9)


def test_same_bar_double_hit_prefers_sl():
    """Cenário C: bar cruza ambos → SL vence (convenção LdP, fill conservador).
    """
    bars = [
        {"open": 100, "high": 100.1, "low": 99.9, "close": 100},
        {"open": 100, "high": 100.2, "low": 99.9, "close": 100.0},  # bar1 (entry)
        {"open": 100.0, "high": 102.5, "low": 97.5, "close": 100.0},  # both crossed
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0},
    ] + [{"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0}] * 6

    ohlc = _build_ohlc(bars)
    close = ohlc["close"]

    events = pd.DataFrame({
        "t1": [close.index[9]],
        "trgt": [0.02],
        "side": [1],
    }, index=[close.index[0]])

    result = apply_triple_barrier(
        close, events, pt_sl=(1.0, 1.0),
        open_prices=ohlc["open"],
        high_prices=ohlc["high"],
        low_prices=ohlc["low"],
    )

    assert len(result) == 1
    assert result.iloc[0]["barrier_type"] == 1  # SL wins on ambiguity
    assert result.iloc[0]["ret"] == pytest.approx(-0.02, abs=1e-9)


def test_breakeven_trigger_on_high_then_sl_same_bar():
    """Cenário D: BE ativa na high (>= upper*be_trigger=0.01) e SL movido (0.0001)
    é atingido pela low na mesma barra. Esperado: barrier_type=sl, ret≈0.0001.
    """
    # Entry bar1 open=100. upper=0.02 (PT @ 2%). be_trigger=0.5 → ativa em +1%.
    # Bar 3: high=101.5 (ativa BE), low=99.5 (<= 0.0001? 99.5/100-1=-0.005, sim).
    bars = [
        {"open": 100, "high": 100.1, "low": 99.9, "close": 100},
        {"open": 100, "high": 100.3, "low": 99.9, "close": 100.2},  # bar1 (entry)
        {"open": 100.2, "high": 100.5, "low": 100.0, "close": 100.3},
        {"open": 100.3, "high": 101.5, "low": 99.5, "close": 100.0},  # BE on, then SL
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0},
    ] + [{"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0}] * 5

    ohlc = _build_ohlc(bars)
    close = ohlc["close"]

    events = pd.DataFrame({
        "t1": [close.index[9]],
        "trgt": [0.02],
        "side": [1],
    }, index=[close.index[0]])

    result = apply_triple_barrier(
        close, events, pt_sl=(1.0, 1.0),
        be_trigger=0.5,
        open_prices=ohlc["open"],
        high_prices=ohlc["high"],
        low_prices=ohlc["low"],
    )

    assert len(result) == 1
    assert result.iloc[0]["barrier_type"] == 1  # sl (post-BE)
    assert result.iloc[0]["ret"] == pytest.approx(0.0001, abs=1e-9)


def test_short_side_pt_hit_by_low():
    """Short: Entry @ 100. PT = +2% side-adjusted → preço cai para 98.
    Bar 3: low 97.5 (cruza PT para short), close 100 (reverte).
    Semântica v2: detecta PT, ret = +0.02 (side-adjusted).
    """
    bars = [
        {"open": 100, "high": 100.1, "low": 99.9, "close": 100},
        {"open": 100, "high": 100.3, "low": 99.7, "close": 100.0},  # bar1 (entry short)
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0},
        {"open": 100.0, "high": 100.1, "low": 97.5, "close": 100.0},  # short PT by low
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0},
    ] + [{"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0}] * 5

    ohlc = _build_ohlc(bars)
    close = ohlc["close"]

    events = pd.DataFrame({
        "t1": [close.index[9]],
        "trgt": [0.02],
        "side": [-1],
    }, index=[close.index[0]])

    result = apply_triple_barrier(
        close, events, pt_sl=(1.0, 1.0),
        open_prices=ohlc["open"],
        high_prices=ohlc["high"],
        low_prices=ohlc["low"],
    )

    assert len(result) == 1
    assert result.iloc[0]["barrier_type"] == 0  # pt
    assert result.iloc[0]["ret"] == pytest.approx(0.02, abs=1e-9)

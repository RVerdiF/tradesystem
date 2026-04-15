"""
Testes unitários para a Barreira Tripla Dinâmica (Breakeven).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.labeling.triple_barrier import create_events, get_labels


def test_breakeven_activation():
    """
    Testa se o breakeven é ativado corretamente.
    Cenário: Preço sobe até atingir o trigger e depois cai até a entrada.
    Entry é na barra T+1 (open_prices[start_loc+1]).
    """
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")

    # Large prices to bypass friction filter
    # Entry em T+1 = open_prices[1] = 100000.0
    # close sobe rápido: atinge 101000.0 (trigger 1%) na barra 5
    # depois cai rápido: atinge 100000.0 (breakeven) na barra 10
    # e continua caindo até 98000.0 (SL original) na barra 15
    close_vals = [100000.0, 100500.0, 101000.0, 101000.0, 100500.0, 101000.0,   # 0-5: sobe
                  100500.0, 100000.0, 99500.0, 99000.0, 98500.0, 98000.0,        # 6-11: cai rápido
                  97500.0, 97000.0, 96500.0, 96000.0, 95500.0, 95000.0,          # 12-17
                  95000.0, 95000.0, 95000.0, 95000.0, 95000.0, 95000.0,          # 18-23
                  95000.0, 95000.0, 95000.0, 95000.0, 95000.0, 95000.0]          # 24-29
    close = pd.Series(close_vals, index=dates)
    open_prices = pd.Series(close_vals, index=dates)

    # Evento na primeira barra (T=0), entrada em T+1 = 100.5 (close[1])
    events_ts = pd.DatetimeIndex([dates[0]])
    # trgt = 0.02 (2%), pt=1.0 (102.0), sl=1.0 (98.0)
    # trigger = 0.5 -> ativa breakeven se lucro >= 1.0%
    targets = pd.Series([0.02], index=events_ts)

    events = create_events(close, events_ts, targets, pt_sl=(1.0, 1.0), max_holding=20)

    # 1. Sem breakeven: SL em 98000.0, close atinge 98000.0 na barra 11 → 'sl'
    labels_no_be = get_labels(close, events, pt_sl=(1.0, 1.0), be_trigger=0.0, open_prices=open_prices, high_prices=close, low_prices=close)
    assert labels_no_be.iloc[0]["barrier_type"] == "sl"

    # 2. Com breakeven: close atinge 101000.0 (ret=1%) na barra 2 → trigger ativado
    # Breakeven move SL para 0.0001, close volta a ~100000.0 na barra 7 → SL hit com ret≈0
    labels_be = get_labels(close, events, pt_sl=(1.0, 1.0), be_trigger=0.5, open_prices=open_prices, high_prices=close, low_prices=close)
    assert labels_be.iloc[0]["barrier_type"] == "sl"
    # Breakeven ativado: SL movido para ~0, retorno próximo de zero (não o SL original de -2%)
    assert abs(labels_be.iloc[0]["ret"]) < 0.03


def test_breakeven_not_activated():
    """Testa se o breakeven NÃO é ativado se o preço não atingir o gatilho."""
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    # Preços: 100000 -> 100800 (não atinge 101000) -> 98000 (SL)
    # trigger = 0.5 de 2% lucro (101000)
    prices = np.linspace(100000, 100800, 10)
    prices = np.append(prices, np.linspace(100800, 97000, 10))
    close = pd.Series(prices, index=dates)
    
    events_ts = pd.DatetimeIndex([dates[0]])
    targets = pd.Series([0.02], index=events_ts)
    
    events = create_events(close, events_ts, targets, pt_sl=(1.0, 1.0), max_holding=20)
    
    labels_be = get_labels(close, events, pt_sl=(1.0, 1.0), be_trigger=0.5, open_prices=close, high_prices=close, low_prices=close)

    # Deve bater no SL original (aprox 98000) pois não atingiu 101000
    assert labels_be.iloc[0]["barrier_type"] == "sl"
    # Retorno deve ser por volta de -2%
    assert labels_be.iloc[0]["ret"] <= -0.019


def test_breakeven_then_tp():
    """Testa breakeven ativado, mas preço continua subindo e atinge TP."""
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    # 100000 -> 103000 (atravessa trigger 101000.0 e TP 102000.0)
    prices = np.linspace(100000, 103000, n)
    close = pd.Series(prices, index=dates)
    
    events_ts = pd.DatetimeIndex([dates[0]])
    targets = pd.Series([0.02], index=events_ts)
    
    events = create_events(close, events_ts, targets, pt_sl=(1.0, 1.0), max_holding=20)
    
    labels_be = get_labels(close, events, pt_sl=(1.0, 1.0), be_trigger=0.5, open_prices=close, high_prices=close, low_prices=close)

    # Deve bater no TP (pt)
    assert labels_be.iloc[0]["barrier_type"] == "pt"
    assert labels_be.iloc[0]["ret"] >= 0.02

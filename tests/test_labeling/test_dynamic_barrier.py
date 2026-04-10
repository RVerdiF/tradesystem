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
    """
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    
    # Preços: 100 -> 101 (trigger 0.5 de 2.0%) -> 100.0 (breakeven hit)
    prices = np.array([
        100.0, 100.2, 100.4, 100.6, 100.8, 101.0,  # Sobe até 101.0 (1% de lucro)
        100.8, 100.6, 100.4, 100.2, 100.0, 99.8,   # Cai de volta
        99.6, 99.4, 99.2, 99.0, 98.8, 98.6, 98.4, 98.2
    ])
    close = pd.Series(prices, index=dates)
    
    # Evento na primeira barra
    events_ts = pd.DatetimeIndex([dates[0]])
    # trgt = 0.02 (2%), pt=1.0 (102.0), sl=1.0 (98.0)
    # trigger = 0.5 -> ativa breakeven se lucro >= 1.0% (101.0)
    targets = pd.Series([0.02], index=events_ts)
    
    events = create_events(close, events_ts, targets, pt_sl=(1.0, 1.0), max_holding=20)
    
    # 1. Sem breakeven: deve ser 'vertical' ou 'sl' original (98.0)
    labels_no_be = get_labels(close, events, pt_sl=(1.0, 1.0), be_trigger=0.0)
    assert labels_no_be.iloc[0]["barrier_type"] in ["vertical", "sl"]
    
    # 2. Com breakeven: deve bater no SL em 100.0 (pois ret 0.0 <= 0.0001)
    labels_be = get_labels(close, events, pt_sl=(1.0, 1.0), be_trigger=0.5)
    assert labels_be.iloc[0]["barrier_type"] == "sl"
    # O retorno deve ser 0.0 (pois o preço voltou exatamente para 100.0)
    assert abs(labels_be.iloc[0]["ret"]) < 1e-6


def test_breakeven_not_activated():
    """Testa se o breakeven NÃO é ativado se o preço não atingir o gatilho."""
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    # Preços: 100 -> 100.8 (não atinge 101.0) -> 98.0 (SL)
    # trigger = 0.5 de 2% lucro (101.0)
    prices = np.linspace(100, 100.8, 10)
    prices = np.append(prices, np.linspace(100.8, 97.0, 10))
    close = pd.Series(prices, index=dates)
    
    events_ts = pd.DatetimeIndex([dates[0]])
    targets = pd.Series([0.02], index=events_ts)
    
    events = create_events(close, events_ts, targets, pt_sl=(1.0, 1.0), max_holding=20)
    
    labels_be = get_labels(close, events, pt_sl=(1.0, 1.0), be_trigger=0.5)
    
    # Deve bater no SL original (aprox 98.0) pois não atingiu 101.0
    assert labels_be.iloc[0]["barrier_type"] == "sl"
    # Retorno deve ser por volta de -2%
    assert labels_be.iloc[0]["ret"] <= -0.019


def test_breakeven_then_tp():
    """Testa breakeven ativado, mas preço continua subindo e atinge TP."""
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    # 100 -> 103 (atravessa trigger 101.0 e TP 102.0)
    prices = np.linspace(100, 103, n)
    close = pd.Series(prices, index=dates)
    
    events_ts = pd.DatetimeIndex([dates[0]])
    targets = pd.Series([0.02], index=events_ts)
    
    events = create_events(close, events_ts, targets, pt_sl=(1.0, 1.0), max_holding=20)
    
    labels_be = get_labels(close, events, pt_sl=(1.0, 1.0), be_trigger=0.5)
    
    # Deve bater no TP (pt)
    assert labels_be.iloc[0]["barrier_type"] == "pt"
    assert labels_be.iloc[0]["ret"] >= 0.02

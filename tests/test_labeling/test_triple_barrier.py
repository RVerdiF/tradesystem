import numpy as np
import pandas as pd
import pytest
from src.labeling.triple_barrier import create_events, apply_triple_barrier, get_labels, _find_dynamic_touch

@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="1d")
    close = pd.Series(np.linspace(100, 200, 100), index=dates)
    
    events_ts = dates[10:90:10]
    target_vol = pd.Series(np.full(len(events_ts), 0.05), index=events_ts)
    side = pd.Series(np.ones(len(events_ts)), index=events_ts)
    
    return close, events_ts, target_vol, side

def test_create_events(sample_data):
    close, events_ts, target_vol, side = sample_data
    events = create_events(close, events_ts, target_vol, side, pt_sl=(1.0, 1.0), max_holding=5)
    
    assert not events.empty
    assert "t1" in events.columns
    assert "trgt" in events.columns
    assert "side" in events.columns
    
def test_create_events_none_side(sample_data):
    close, events_ts, target_vol, _ = sample_data
    events = create_events(close, events_ts, target_vol, side=None, pt_sl=(1.0, 1.0), max_holding=5)
    
    assert (events["side"] == 1).all()

def test_apply_triple_barrier(sample_data):
    close, events_ts, target_vol, side = sample_data
    events = create_events(close, events_ts, target_vol, side, pt_sl=(1.0, 1.0), max_holding=5)
    
    result = apply_triple_barrier(close, events)
    assert not result.empty
    assert "barrier_type" in result.columns

def test_apply_triple_barrier_breakeven(sample_data):
    close, events_ts, target_vol, side = sample_data
    events = create_events(close, events_ts, target_vol, side, pt_sl=(2.0, 1.0), max_holding=20)
    
    # Preços oscilantes para triggar breakeven
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100))
    # Força subida depois queda
    prices[11:15] += 10
    prices[15:20] -= 20
    close = pd.Series(prices, index=close.index)
    
    result = apply_triple_barrier(close, events, be_trigger=0.5)
    assert not result.empty

def test_get_labels(sample_data):
    close, events_ts, target_vol, side = sample_data
    events = create_events(close, events_ts, target_vol, side, pt_sl=(1.0, 1.0), max_holding=5)
    
    labels = get_labels(close, events)
    assert not labels.empty
    assert "label" in labels.columns
    
def test_get_labels_empty():
    close = pd.Series(dtype=float)
    events = pd.DataFrame(columns=["t1", "trgt", "side"])
    labels = get_labels(close, events)
    assert labels.empty


import numpy as np
import pandas as pd
import pytest

from src.labeling.triple_barrier import apply_triple_barrier, create_events, get_labels


@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="1d")
    close = pd.Series(np.linspace(100000, 200000, 100), index=dates)

    events_ts = dates[10:90:10]
    # Make target volatility higher (e.g. 1%) so that profit target > cost.
    target_vol = pd.Series(np.full(len(events_ts), 0.01), index=events_ts)
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

    # open_prices é obrigatório — usa close como synthetic open (explícito, auditável)
    result = apply_triple_barrier(
        close, events, open_prices=close, high_prices=close, low_prices=close
    )
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

    result = apply_triple_barrier(
        close, events, be_trigger=0.5, open_prices=close, high_prices=close, low_prices=close
    )
    assert not result.empty


def test_get_labels(sample_data):
    close, events_ts, target_vol, side = sample_data
    events = create_events(close, events_ts, target_vol, side, pt_sl=(1.0, 1.0), max_holding=5)

    labels = get_labels(close, events, open_prices=close, high_prices=close, low_prices=close)
    assert not labels.empty
    assert "label" in labels.columns


def test_get_labels_empty():
    # No OHLC args needed: the len(close)==0 short-circuit in get_labels fires
    # before the high_prices/low_prices validation assertions are reached.
    # If the short-circuit is ever removed, this test will raise AssertionError
    # (not TypeError) — that is the intended failure signal.
    close = pd.Series(dtype=float)
    events = pd.DataFrame(columns=["t1", "trgt", "side"])
    labels = get_labels(close, events)
    assert labels.empty


def test_entry_price_is_open_t1(sample_data):
    """Entry price must be open[t+1], not close[t]. Regression guard for lookahead bias.

    Uses synthetic OHLC (open=close, high=close+5, low=close-5) because sample_data
    does not provide separate OHLCV columns. The regression property holds regardless:
    if apply_triple_barrier uses open[t+1] as entry, inflating open[t+1] by 10% must
    change the returned ret. If it used close[t], the result would be unchanged.
    """
    close, events_ts, target_vol, side = sample_data
    open_prices = close.copy()  # synthetic: distinct open values introduced by shift below
    high = close.copy() + 5
    low = close.copy() - 5

    events = create_events(close, events_ts, target_vol, side, pt_sl=(2.0, 2.0), max_holding=10)
    result = apply_triple_barrier(
        close, events, pt_sl=(2.0, 2.0), open_prices=open_prices, high_prices=high, low_prices=low
    )

    open_shifted = open_prices.copy()
    first_event_idx = events.index[0]
    t1_loc = close.index.get_loc(first_event_idx) + 1
    t1_idx = close.index[t1_loc]
    open_shifted.loc[t1_idx] = open_shifted.loc[t1_idx] * 1.10

    result_shifted = apply_triple_barrier(
        close, events, pt_sl=(2.0, 2.0), open_prices=open_shifted, high_prices=high, low_prices=low
    )
    assert not result.empty
    assert not result["ret"].equals(result_shifted["ret"]), (
        "Entry price is not using open[t+1] — result is unchanged after modifying open[t+1]."
    )

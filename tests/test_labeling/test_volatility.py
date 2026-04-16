import numpy as np
import pandas as pd
import pytest

from src.labeling.volatility import daily_vol, get_volatility_targets


@pytest.fixture
def sample_close():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="1d")
    returns = np.random.normal(0.001, 0.02, 100)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates)


def test_daily_vol(sample_close):
    vol = daily_vol(sample_close, span=10, log_returns=True)
    assert not vol.empty
    assert vol.index.equals(sample_close.index)
    assert not vol.isna().all()


def test_daily_vol_simple_returns(sample_close):
    vol = daily_vol(sample_close, span=10, log_returns=False)
    assert not vol.empty


def test_daily_vol_none_span(sample_close):
    vol = daily_vol(sample_close, span=None)
    assert not vol.empty


def test_get_volatility_targets(sample_close):
    events = sample_close.index[::5]
    targets = get_volatility_targets(sample_close, events, span=10)
    assert not targets.empty
    assert len(targets) <= len(events)
    assert targets.name == "target"


def test_get_volatility_targets_nan_handling(sample_close):
    # Ensure there are NaNs at the beginning by taking events from the start
    events = sample_close.index[:5]
    targets = get_volatility_targets(sample_close, events, span=20)
    assert len(targets) < len(events)  # some removed due to nan

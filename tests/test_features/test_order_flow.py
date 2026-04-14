import pandas as pd
import numpy as np
import pytest
from src.features.order_flow import compute_vir, tick_rule_direction

def test_tick_rule_direction():
    close = pd.Series([10, 11, 11, 10, 10, 12])
    # diff: [NaN, 1, 0, -1, 0, 2]
    # expected: [0, 1, 1, -1, -1, 1]
    expected = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    result = tick_rule_direction(close)
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_compute_vir():
    df = pd.DataFrame({
        "close": [10, 11, 11, 10, 10, 12],
        "volume": [100, 100, 100, 100, 100, 100]
    })
    # direction: [0, 1, 1, -1, -1, 1]
    # vol: [100, 100, 100, 100, 100, 100]
    # signed_vol: [0, 100, 100, -100, -100, 100]
    # buy_vol: [0, 100, 100, 0, 0, 100]
    # sell_vol: [0, 0, 0, 100, 100, 0]

    # window = 2
    # rolling_buy: [0, 100, 200, 100, 0, 100]
    # rolling_sell: [0, 0, 0, 100, 200, 100]
    # rolling_total: [100, 200, 200, 200, 200, 200]

    # vir: [0/100, 100/200, 200/200, (100-100)/200, (0-200)/200, (100-100)/200]
    # vir: [0.0, 0.5, 1.0, 0.0, -1.0, 0.0]

    expected = pd.Series([0.0, 0.5, 1.0, 0.0, -1.0, 0.0], name="vir")
    result = compute_vir(df, window=2)
    pd.testing.assert_series_equal(result, expected)

def test_compute_vir_zero_vol():
    df = pd.DataFrame({
        "close": [10, 11],
        "volume": [0, 0]
    })
    result = compute_vir(df, window=2)
    assert np.isnan(result).all()

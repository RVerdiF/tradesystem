import pandas as pd

from src.features.order_flow import tick_rule_direction


def test_tick_rule_direction():
    close = pd.Series([10, 11, 11, 10, 10, 12])
    # diff: [NaN, 1, 0, -1, 0, 2]
    # expected: [0, 1, 1, -1, -1, 1]
    expected = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    result = tick_rule_direction(close)
    pd.testing.assert_series_equal(result, expected, check_names=False)

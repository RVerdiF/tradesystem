import math
import pytest
from src.optimization.tuner import _sharpe_lift_multiplier, _filter_rate_multiplier

def test_sharpe_lift_multiplier_zero_lift():
    assert _sharpe_lift_multiplier(0.0) == 1.0

def test_sharpe_lift_multiplier_negative():
    assert _sharpe_lift_multiplier(-1.0) == pytest.approx(math.exp(-2.0), rel=1e-6)

def test_filter_rate_below_soft_start():
    assert _filter_rate_multiplier(0.65) == 1.0

def test_filter_rate_mid_band():
    mult = _filter_rate_multiplier(0.85)
    assert mult is not None
    assert 0.4 < mult < 0.7

def test_filter_rate_hard_reject():
    assert _filter_rate_multiplier(0.93) is None

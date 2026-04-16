import pandas as pd
import pytest

from src.modeling.bet_sizing import compute_kelly_fraction


def test_compute_kelly_fraction():
    prob = 0.6
    win_rate = 1.0  # 1:1 pt/sl ratio
    kelly = compute_kelly_fraction(prob, win_rate, fraction=1.0)
    # p - q = 0.6 - 0.4 = 0.2
    assert kelly == pytest.approx(0.2)

    kelly_half = compute_kelly_fraction(prob, win_rate, fraction=0.5)
    assert kelly_half == pytest.approx(0.1)


def test_compute_kelly_fraction_loss():
    prob = 0.4
    win_rate = 1.0
    kelly = compute_kelly_fraction(prob, win_rate)
    assert kelly == 0.0


def test_compute_kelly_fraction_series():
    prob = pd.Series([0.4, 0.6, 0.8, 1.0])
    kelly = compute_kelly_fraction(prob, fraction=1.0)
    assert (kelly >= 0).all() and (kelly <= 1).all()
    assert kelly.iloc[0] == 0.0  # prob <= 0.5
    assert kelly.iloc[-1] == 1.0  # prob = 1.0


def test_compute_kelly_fraction_none_params():
    kelly = compute_kelly_fraction(0.6, fraction=None)
    assert isinstance(kelly, float)

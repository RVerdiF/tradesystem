import numpy as np
import pandas as pd
import pytest
from src.modeling.bet_sizing import compute_kelly_fraction, discretize_bet, apply_conviction_threshold

def test_compute_kelly_fraction():
    prob = 0.6
    win_rate = 1.0 # 1:1 pt/sl ratio
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
    assert kelly.iloc[0] == 0.0 # prob <= 0.5
    assert kelly.iloc[-1] == 1.0 # prob = 1.0

def test_discretize_bet():
    kelly = pd.Series([0.0, 0.2, 0.5, 0.8, 1.0])
    bets = discretize_bet(kelly, max_position=10, step_size=1)
    assert (bets == [0, 2, 5, 8, 10]).all()

def test_discretize_bet_max_pos_zero():
    kelly = pd.Series([0.5])
    bets = discretize_bet(kelly, max_position=0)
    assert (bets == [0]).all()

def test_discretize_bet_step_size():
    kelly = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
    bets = discretize_bet(kelly, max_position=10, step_size=5)
    assert (bets == [0, 0, 5, 10, 10]).all()

def test_apply_conviction_threshold_scalar():
    assert apply_conviction_threshold(0.4, threshold=0.5) == 0.0
    assert apply_conviction_threshold(0.6, threshold=0.5) == 0.6

def test_apply_conviction_threshold_series():
    probs = pd.Series([0.3, 0.5, 0.7])
    filtered = apply_conviction_threshold(probs, threshold=0.5)
    assert (filtered == [0.0, 0.5, 0.7]).all()

def test_apply_conviction_threshold_ndarray():
    probs = np.array([0.3, 0.5, 0.7])
    filtered = apply_conviction_threshold(probs, threshold=0.5)
    assert (filtered == [0.0, 0.5, 0.7]).all()

def test_compute_kelly_fraction_none_params():
    kelly = compute_kelly_fraction(0.6, fraction=None)
    assert isinstance(kelly, float)

def test_discretize_bet_none_params():
    kelly = pd.Series([0.5])
    bets = discretize_bet(kelly, max_position=None)
    assert isinstance(bets, pd.Series)

def test_apply_conviction_threshold_none_params():
    filtered = apply_conviction_threshold(0.6, threshold=None)
    assert isinstance(filtered, float)

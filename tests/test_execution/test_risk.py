import datetime

import pytest

from src.execution.risk import STATE_HALTED_FOR_DAY, RiskManager


@pytest.fixture
def risk_manager():
    return RiskManager(start_balance=10000.0)


def test_can_trade_system_halted(risk_manager):
    """Test that can_trade returns False when the system is halted."""
    risk_manager._set_state(STATE_HALTED_FOR_DAY, "test")
    assert not risk_manager.can_trade()


def test_can_trade_no_restrictions(risk_manager):
    """Test that can_trade returns True when there are no restrictions."""
    assert risk_manager.can_trade("EURUSD")


def test_can_trade_active_cooldown(risk_manager):
    """Test that can_trade returns False when there is an active cool-down for the symbol."""
    future_time = datetime.datetime.now() + datetime.timedelta(minutes=5)
    risk_manager._cool_down_until["EURUSD"] = future_time
    assert not risk_manager.can_trade("EURUSD")


def test_can_trade_expired_cooldown(risk_manager):
    """Test that can_trade returns True and removes the cool-down when it has expired."""
    past_time = datetime.datetime.now() - datetime.timedelta(minutes=5)
    risk_manager._cool_down_until["EURUSD"] = past_time
    assert risk_manager.can_trade("EURUSD")
    assert "EURUSD" not in risk_manager._cool_down_until

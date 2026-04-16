import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.execution.engine import AsyncTradingEngine
from src.execution.risk import RiskManager


class TestTimeRestrictions:
    def test_risk_manager_time_window(self):
        """Valida se o RiskManager bloqueia/libera baseado no horário."""
        # Config: 09:00 to 17:55
        rm = RiskManager(
            start_balance=100000.0,
            trade_type="day_trade",
            start_time="09:00:00",
            end_time="17:55:00",
        )

        # 1. Antes do horário (08:59)
        with patch("src.execution.risk.datetime.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = datetime.time(8, 59, 0)
            mock_dt.date = datetime.date

            rm.update_equity(100000.0, 100000.0)
            assert rm.can_trade() is False
            assert "OUTSIDE TRADING WINDOW" in rm.halt_reason

        # 2. Dentro do horário (09:01)
        with patch("src.execution.risk.datetime.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = datetime.time(9, 1, 0)
            mock_dt.date = datetime.date

            rm.update_equity(100000.0, 100000.0)
            assert rm.can_trade() is True
            assert rm.halt_reason == ""

        # 3. Depois do horário (17:56)
        with patch("src.execution.risk.datetime.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = datetime.time(17, 56, 0)
            mock_dt.date = datetime.date

            rm.update_equity(100000.0, 100000.0)
            assert rm.can_trade() is False
            assert "OUTSIDE TRADING WINDOW" in rm.halt_reason

    @pytest.mark.asyncio
    async def test_engine_day_trade_closes_positions(self):
        """Valida se o Engine fecha posições no Day Trade ao bater o horário."""
        mock_pipeline = MagicMock(return_value={"side": 0})
        symbols = ["PETR4"]

        # Configurado como Day Trade
        engine = AsyncTradingEngine(mock_pipeline, symbols, trade_type="day_trade")

        # Mock do OrderManager para verificar se close_positions foi chamado
        engine.om = MagicMock()
        engine.om.get_net_position.return_value = 1.0  # Tem posição aberta

        # Força o RiskManager a estar haltado por horário
        engine.risk.is_halted = True
        engine.risk.halt_reason = "OUTSIDE TRADING WINDOW (17:56:00)"

        await engine._process_symbol("PETR4")

        # Deve ter chamado close_positions
        engine.om.close_positions.assert_called_once_with("PETR4")

    @pytest.mark.asyncio
    async def test_engine_swing_trade_keeps_positions(self):
        """Valida se o Engine MANTÉM posições no Swing Trade fora do horário."""
        mock_pipeline = MagicMock(return_value={"side": 0})
        symbols = ["PETR4"]

        # Configurado como Swing Trade
        engine = AsyncTradingEngine(mock_pipeline, symbols, trade_type="swing_trade")

        # Mock do OrderManager
        engine.om = MagicMock()
        engine.om.get_net_position.return_value = 1.0  # Tem posição aberta

        # Força o RiskManager a estar haltado por horário
        engine.risk.is_halted = True
        engine.risk.halt_reason = "OUTSIDE TRADING WINDOW (18:00:00)"

        await engine._process_symbol("PETR4")

        # NÃO deve ter chamado close_positions
        engine.om.close_positions.assert_not_called()

    @pytest.mark.asyncio
    async def test_engine_swing_trade_closes_on_pnl_halt(self):
        """Valida se o Engine FECHA posições no Swing Trade se o motivo for PnL."""
        mock_pipeline = MagicMock(return_value={"side": 0})
        symbols = ["PETR4"]

        engine = AsyncTradingEngine(mock_pipeline, symbols, trade_type="swing_trade")
        engine.om = MagicMock()
        engine.om.get_net_position.return_value = 1.0

        # Força o RiskManager a estar haltado por PERDA DIÁRIA (não horário)
        engine.risk.is_halted = True
        engine.risk.halt_reason = "MAX DAILY LOSS REACHED"

        await engine._process_symbol("PETR4")

        # DEVE ter chamado close_positions
        engine.om.close_positions.assert_called_once_with("PETR4")


class TestCoolDown:
    def test_cool_down_activated_after_notify(self):
        """notify_trade_closed() deve activar o cool-down e bloquear novas ordens."""
        import src.execution.risk as risk_module

        rm = RiskManager(start_balance=100000.0)

        # Garante que está ACTIVE dentro do horário
        mock_now = datetime.datetime(2026, 4, 12, 10, 0, 0)
        with patch.object(risk_module.datetime, "datetime", wraps=datetime.datetime) as mock_dt_cls:
            mock_dt_cls.now.return_value = mock_now
            rm.update_equity(100000.0, 100000.0)

        assert rm.can_trade("PETR4") is True

        # Activa o cool-down (simula saída por circuit breaker)
        with patch.object(risk_module.datetime, "datetime", wraps=datetime.datetime) as mock_dt_cls:
            mock_dt_cls.now.return_value = mock_now
            rm.notify_trade_closed("PETR4")
            assert rm.can_trade("PETR4") is False

        assert "PETR4" in rm._cool_down_until

    def test_cool_down_expires_after_time(self):
        """Após o temporizador expirar, update_equity() deve reativar o sistema."""
        import src.execution.risk as risk_module

        rm = RiskManager(start_balance=100000.0)

        # Activa cool-down às 10:00:00
        mock_now_1 = datetime.datetime(2026, 4, 12, 10, 0, 0)
        with patch.object(risk_module.datetime, "datetime", wraps=datetime.datetime) as mock_dt_cls:
            mock_dt_cls.now.return_value = mock_now_1
            rm.notify_trade_closed("PETR4")

        assert "PETR4" in rm._cool_down_until

        # Simula tick às 10:06:00 (após 5 min de cool-down)
        future = datetime.datetime(2026, 4, 12, 10, 6, 0)
        with patch.object(risk_module.datetime, "datetime", wraps=datetime.datetime) as mock_dt_cls:
            mock_dt_cls.now.return_value = future
            rm.update_equity(100000.0, 100000.0)
            assert rm.can_trade("PETR4") is True

        assert "PETR4" not in rm._cool_down_until
        assert rm.system_state == "ACTIVE"
        assert rm.halt_reason == ""

    def test_daily_profit_target_halts_trading(self):
        """Se o PnL diário atingir max_daily_profit_pct, o sistema deve parar."""
        rm = RiskManager(start_balance=100000.0)

        mock_now = MagicMock()
        mock_now.time.return_value = datetime.time(10, 0, 0)
        with patch("src.execution.risk.datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.date = datetime.date
            mock_dt.timedelta = datetime.timedelta
            # Equity = 102001 sobre start_balance 100000 → +2.001% > limite de 2%
            rm.update_equity(100000.0, 102001.0)

        assert rm.can_trade() is False
        assert rm.system_state == "HALTED_FOR_DAY"

    def test_daily_profit_target_in_halt_reason(self):
        """O motivo do halt deve identificar claramente a meta de lucro atingida."""
        rm = RiskManager(start_balance=100000.0)

        mock_now = MagicMock()
        mock_now.time.return_value = datetime.time(10, 0, 0)
        with patch("src.execution.risk.datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.date = datetime.date
            mock_dt.timedelta = datetime.timedelta
            rm.update_equity(100000.0, 102001.0)

        assert "MAX DAILY PROFIT REACHED" in rm.halt_reason

    def test_cool_down_does_not_override_halted_for_day(self):
        """notify_trade_closed() não deve sobrescrever um HALTED_FOR_DAY permanente."""
        rm = RiskManager(start_balance=100000.0)

        # Força estado HALTED_FOR_DAY via perda diária
        mock_now = MagicMock()
        mock_now.time.return_value = datetime.time(10, 0, 0)
        with patch("src.execution.risk.datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.date = datetime.date
            mock_dt.timedelta = datetime.timedelta
            rm.update_equity(100000.0, 97000.0)  # -3% > limite de 2%

        assert rm.system_state == "HALTED_FOR_DAY"

        # Tenta activar cool-down — deve ser ignorado
        with patch("src.execution.risk.datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.date = datetime.date
            mock_dt.timedelta = datetime.timedelta
            rm.notify_trade_closed("PETR4")

        # Estado não deve ter mudado e cool-down dictionary deve continuar vazio
        assert rm.system_state == "HALTED_FOR_DAY"
        assert "MAX DAILY LOSS REACHED" in rm.halt_reason
        assert "PETR4" not in rm._cool_down_until

    def test_validate_order_halted(self):
        """Testa validate_order quando não pode operar"""
        rm = RiskManager(start_balance=100000.0)
        rm.is_halted = True
        assert rm.validate_order(1.0, 1.0, 5.0) is False

    def test_validate_order_max_position(self):
        """Testa validate_order acima do máximo"""
        rm = RiskManager(start_balance=100000.0)
        assert rm.validate_order(1.0, 5.0, 5.0) is False

    def test_validate_order_valid(self):
        """Testa validate_order valido"""
        rm = RiskManager(start_balance=100000.0)
        assert rm.validate_order(1.0, 1.0, 5.0) is True

    def test_notify_trade_closed_outside_window(self):
        """Testa notify trade closed ignores outside window."""
        rm = RiskManager(start_balance=100000.0)
        rm.system_state = "OUTSIDE_WINDOW"
        rm.notify_trade_closed("PETR4")
        assert rm.system_state == "OUTSIDE_WINDOW"
        assert "PETR4" not in rm._cool_down_until

    def test_notify_trade_closed_no_cooldown(self):
        """Testa notify trade closed ignores if no cooldown setup."""
        rm = RiskManager(start_balance=100000.0)
        rm.cool_down_minutes = 0
        rm.notify_trade_closed("PETR4")
        assert rm.system_state == "ACTIVE"
        assert "PETR4" not in rm._cool_down_until


class TestUpdateEquity:
    def test_update_equity_daily_reset(self):
        """Valida se a mudança de dia reseta corretamente saldo, estado e cool-downs."""
        rm = RiskManager(start_balance=100000.0)

        # Estado anterior
        rm.system_state = "HALTED_FOR_DAY"
        rm.halt_reason = "MAX DAILY LOSS REACHED"
        rm.last_trading_day = datetime.date(2024, 1, 1)
        rm._cool_down_until["PETR4"] = datetime.datetime.now() + datetime.timedelta(minutes=30)

        with patch("src.execution.risk.datetime") as mock_datetime:
            mock_datetime.date.today.return_value = datetime.date(2024, 1, 2)
            mock_now = MagicMock()
            mock_now.time.return_value = datetime.time(10, 0, 0)  # Dentro do horário
            mock_datetime.datetime.now.return_value = mock_now

            # Envia novo saldo
            rm.update_equity(105000.0, 105000.0)

        # Validações do Reset Diário
        assert rm.start_balance == 105000.0
        assert rm.last_trading_day == datetime.date(2024, 1, 2)
        assert len(rm._cool_down_until) == 0
        assert rm.system_state == "ACTIVE"
        assert rm.halt_reason == ""

    def test_update_equity_missing_start_balance(self):
        """Testa se balance é inicializado quando for None."""
        import src.execution.risk as risk_module

        rm = RiskManager(start_balance=None)

        with patch.object(risk_module.datetime, "date", wraps=datetime.date) as mock_date:
            mock_date.today.return_value = datetime.date.today()
            rm.last_trading_day = datetime.date.today()

            with patch("src.execution.risk.datetime.datetime") as mock_dt:
                mock_now = MagicMock()
                mock_now.time.return_value = datetime.time(10, 0, 0)
                mock_dt.now.return_value = mock_now

                rm.update_equity(100000.0, 100000.0)

        assert rm.start_balance == 100000.0
        assert rm.highest_equity == 100000.0

    def test_update_equity_halted_for_day_returns_early(self):
        """Testa se _check_circuit_breakers retorna imediatamente quando STATE_HALTED_FOR_DAY."""
        rm = RiskManager(start_balance=100000.0)
        rm._set_state("HALTED_FOR_DAY", "SOME REASON")

        mock_now = MagicMock()
        mock_now.time.return_value = datetime.time(8, 0, 0)  # Fora do horário
        with patch("src.execution.risk.datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.date = datetime.date
            rm.update_equity(100000.0, 100000.0)

        assert rm.system_state == "HALTED_FOR_DAY"
        assert rm.halt_reason == "SOME REASON"

    def test_max_drawdown_halts_trading(self):
        """Testa se _check_circuit_breakers paralisa o sistema em max drawdown."""
        rm = RiskManager(start_balance=100000.0)
        rm.max_daily_profit_pct = 0.20

        mock_now = MagicMock()
        mock_now.time.return_value = datetime.time(10, 0, 0)
        with patch("src.execution.risk.datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.date = datetime.date

            rm.update_equity(100000.0, 110000.0)
            assert rm.highest_equity == 110000.0

            rm.update_equity(100000.0, 104000.0)

        assert rm.system_state == "HALTED_FOR_DAY"
        assert "MAX DRAWDOWN REACHED" in rm.halt_reason

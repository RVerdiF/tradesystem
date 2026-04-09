import pytest
import datetime
import asyncio
from unittest.mock import patch, MagicMock
from src.execution.risk import RiskManager
from src.execution.engine import AsyncTradingEngine

class TestTimeRestrictions:
    
    def test_risk_manager_time_window(self):
        """Valida se o RiskManager bloqueia/libera baseado no horário."""
        # Config: 09:00 to 17:55
        rm = RiskManager(
            start_balance=100000.0, 
            trade_type="day_trade",
            start_time="09:00:00", 
            end_time="17:55:00"
        )
        
        # 1. Antes do horário (08:59)
        with patch('src.execution.risk.datetime.datetime') as mock_dt:
            mock_dt.now.return_value.time.return_value = datetime.time(8, 59, 0)
            
            rm.update_equity(100000.0, 100000.0)
            assert rm.can_trade() is False
            assert "OUTSIDE TRADING WINDOW" in rm.halt_reason
            
        # 2. Dentro do horário (09:01)
        with patch('src.execution.risk.datetime.datetime') as mock_dt:
            mock_dt.now.return_value.time.return_value = datetime.time(9, 1, 0)
            
            rm.update_equity(100000.0, 100000.0)
            assert rm.can_trade() is True
            assert rm.halt_reason == ""
            
        # 3. Depois do horário (17:56)
        with patch('src.execution.risk.datetime.datetime') as mock_dt:
            mock_dt.now.return_value.time.return_value = datetime.time(17, 56, 0)
            
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
        engine.om.get_net_position.return_value = 1.0 # Tem posição aberta
        
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
        engine.om.get_net_position.return_value = 1.0 # Tem posição aberta
        
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

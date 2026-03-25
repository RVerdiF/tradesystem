"""
Testes para a Fase 6 — Paper Trading e Implantação.

Testa isoladamente:
- RiskManager (limites de PnL e Drawdown)
- OrderManager (Sinalização via Mock do MT5)
- Audit (escruta correta de JSONL)
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.execution.risk import RiskManager
from src.execution.audit import AuditLogger
from src.execution.order_manager import OrderManager


# ---------------------------------------------------------------------------
# Testes — Auditoria
# ---------------------------------------------------------------------------
class TestAudit:

    def test_log_signal_creates_jsonl(self):
        """Auditor deve criar um JSON válido."""
        with TemporaryDirectory() as tmpdir:
            temp_audit = AuditLogger(log_dir=tmpdir)
            
            temp_audit.log_signal("WINZ25", alpha_side=1, meta_label=1, kelly_fraction=0.5, price=105000.0)
            
            # Lê o JSONL gerado
            with open(temp_audit.signal_log_file, "r") as f:
                lines = f.readlines()
            
            assert len(lines) == 1
            data = json.loads(lines[0])
            
            assert data["event"] == "SIGNAL_GENERATED"
            assert data["symbol"] == "WINZ25"
            assert data["meta_label"] == 1
            assert "timestamp" in data

    def test_log_order_creates_jsonl(self):
        with TemporaryDirectory() as tmpdir:
            temp_audit = AuditLogger(log_dir=tmpdir)
            temp_audit.log_order(12345, "PETR4", "buy", 100, 35.50, "Test Order")
            
            with open(temp_audit.trade_log_file, "r") as f:
                lines = f.readlines()
                
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["action"] == "buy"
            assert data["ticket"] == 12345


# ---------------------------------------------------------------------------
# Testes — Risk Manager (Circuit Breakers)
# ---------------------------------------------------------------------------
class TestRiskManager:

    def test_initial_state_can_trade(self):
        risk = RiskManager(start_balance=10000.0)
        assert risk.can_trade() is True
        assert risk.is_halted is False

    def test_daily_loss_circuit_breaker(self):
        """Se o PnL diário cair abaixo do max_daily_loss, deve travar."""
        risk = RiskManager(start_balance=10000.0)
        risk.max_daily_loss_pct = 0.05  # -5%
        
        # Perde 3% -> OK
        risk.update_equity(balance=10000.0, equity=9700.0)
        assert risk.can_trade() is True
        
        # Perde 6% -> BREAK
        risk.update_equity(balance=10000.0, equity=9300.0)
        assert risk.can_trade() is False
        assert "MAX DAILY LOSS" in risk.halt_reason

    def test_drawdown_circuit_breaker(self):
        """Drawdown em relação à máxima histórica."""
        risk = RiskManager(start_balance=10000.0)
        risk.max_drawdown_pct = 0.10  # -10% MDD
        
        # Sobe pra 20.000
        risk.update_equity(balance=15000.0, equity=20000.0)
        assert risk.can_trade() is True
        
        # Cai 5% (de 20.000 = 19.000)
        risk.update_equity(balance=15000.0, equity=19000.0)
        assert risk.can_trade() is True
        
        # Cai 11% (de 20.000 = 17.800)
        risk.update_equity(balance=15000.0, equity=17800.0)
        assert risk.can_trade() is False
        assert "MAX DRAWDOWN" in risk.halt_reason

    def test_validate_order_exposure(self):
        """Impede ordens que excedam a mão máxima parametrizada."""
        risk = RiskManager(start_balance=10000.0)
        
        # Tem 2 lotes, quer abrir mais 4 limitando em 5 = BLOCK
        assert risk.validate_order(current_exposure=2.0, new_volume=4.0, max_exposure=5.0) is False
        
        # Tem 2 lotes, quer abrir mais 3 limitando em 5 = PASS
        assert risk.validate_order(current_exposure=2.0, new_volume=3.0, max_exposure=5.0) is True


# ---------------------------------------------------------------------------
# Testes — Order Manager (Mock)
# ---------------------------------------------------------------------------
class TestOrderManager:
    
    def test_paper_mode_bypasses_mt5(self):
        """No modo paper, send_market_order apenas loga e não chama a DLL mt5."""
        om = OrderManager()
        # Mock do execution_config via patch nao necessario se default é paper
        success = om.send_market_order("WIN", "buy", 1.0)
        assert success is True
        
        om.close_positions("WIN") # Nao falha sem MT5 ligado se for paper

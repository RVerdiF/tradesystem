"""Testes para a Fase 6 — Paper Trading e Implantação.

Testa isoladamente:
- AuditLogger (persistência correta em SQLite)
- RiskManager (limites de PnL e Drawdown)
- OrderManager (Sinalização via Mock do MT5)
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch, MagicMock

import pytest

from src.execution.risk import RiskManager
from src.execution.audit import AuditLogger
from src.execution.order_manager import OrderManager
from src.db import _ALL_DDL


# ---------------------------------------------------------------------------
# Fixture: banco SQLite isolado por teste
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_audit(tmp_path):
    """Cria um AuditLogger com banco SQLite temporário isolado por teste.
    Patcha get_connection para apontar ao banco temp em vez do real.
    """
    db_file = tmp_path / "test_audit.db"

    def _temp_conn():
        conn = sqlite3.connect(str(db_file))
        conn.row_factory = sqlite3.Row
        for ddl in _ALL_DDL:
            conn.execute(ddl)
        conn.commit()
        return conn

    with (
        patch("src.execution.audit.get_connection", side_effect=_temp_conn),
        patch("src.execution.audit.init_db"),
    ):
        audit = AuditLogger()
        audit._get_conn = _temp_conn  # para que os testes possam consultar
        yield audit, _temp_conn


# ---------------------------------------------------------------------------
# Testes — Auditoria (SQLite)
# ---------------------------------------------------------------------------

class TestAudit:

    def test_log_signal_persisted_in_db(self, isolated_audit):
        """log_signal deve inserir um registro válido na tabela audit_signals."""
        audit, get_conn = isolated_audit

        audit.log_signal("WINZ25", alpha_side=1, meta_label=1, kelly_fraction=0.5, price=105000.0)

        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_signals WHERE symbol = 'WINZ25'"
            ).fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row["symbol"] == "WINZ25"
        assert row["alpha_side"] == 1
        assert row["meta_label"] == 1
        assert row["kelly_fraction"] == pytest.approx(0.5)
        assert row["reference_price"] == pytest.approx(105000.0)
        assert row["timestamp"] != ""

    def test_log_order_persisted_in_db(self, isolated_audit):
        """log_order deve inserir um registro válido na tabela audit_orders."""
        audit, get_conn = isolated_audit

        audit.log_order(12345, "PETR4", "buy", 100, 35.50, "Test Order")

        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_orders WHERE symbol = 'PETR4'"
            ).fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row["ticket"] == 12345
        assert row["action"] == "buy"
        assert row["volume"] == pytest.approx(100.0)
        assert row["price"] == pytest.approx(35.50)
        assert row["comment"] == "Test Order"

    def test_log_error_non_critical(self, isolated_audit):
        """log_error com critical=False deve persistir critical=0."""
        audit, get_conn = isolated_audit

        audit.log_error("TestComponent", "Minor issue", critical=False)

        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_errors WHERE component = 'TestComponent'"
            ).fetchall()

        assert len(rows) == 1
        assert rows[0]["critical"] == 0
        assert rows[0]["error_msg"] == "Minor issue"

    def test_log_error_critical_flag(self, isolated_audit):
        """log_error com critical=True deve persistir critical=1."""
        audit, get_conn = isolated_audit

        audit.log_error("RiskManager", "MAX DAILY LOSS REACHED", critical=True)

        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_errors WHERE critical = 1"
            ).fetchall()

        assert len(rows) == 1
        assert "MAX DAILY LOSS" in rows[0]["error_msg"]

    def test_query_signals_by_symbol(self, isolated_audit):
        """query_signals deve filtrar por symbol corretamente."""
        audit, _ = isolated_audit

        audit.log_signal("WINZ25", 1, 1, 0.5, 100.0)
        audit.log_signal("PETR4", -1, 0, 0.0, 35.0)

        signals = audit.query_signals(symbol="WINZ25")
        assert len(signals) == 1
        assert signals[0]["symbol"] == "WINZ25"

    def test_query_orders_by_symbol(self, isolated_audit):
        """query_orders deve filtrar por symbol corretamente."""
        audit, _ = isolated_audit

        audit.log_order(1, "WINZ25", "buy", 1.0, 100.0)
        audit.log_order(2, "PETR4", "sell", 2.0, 35.0)

        orders = audit.query_orders(symbol="PETR4")
        assert len(orders) == 1
        assert orders[0]["ticket"] == 2

    def test_query_errors_critical_only(self, isolated_audit):
        """query_errors com critical_only=True deve retornar apenas erros críticos."""
        audit, _ = isolated_audit

        audit.log_error("A", "normal", critical=False)
        audit.log_error("B", "critical one", critical=True)

        critical = audit.query_errors(critical_only=True)
        assert len(critical) == 1
        assert critical[0]["error_msg"] == "critical one"


# ---------------------------------------------------------------------------
# Testes — Risk Manager (Circuit Breakers)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_trading_context():
    """Garante que os testes rodam sempre dentro do horário e em modo paper."""
    import datetime
    from unittest.mock import MagicMock
    with (
        patch("src.execution.risk.datetime.datetime") as mock_dt,
        patch("src.execution.order_manager.execution_config") as mock_exec,
        patch("src.execution.risk.risk_config") as mock_risk
    ):
        # Sempre 10:00 da manhã
        mock_dt.now.return_value.time.return_value = datetime.time(10, 0, 0)
        
        # Modo Paper por padrão nos testes
        mock_exec.mode = "paper"
        
        # Configurações de risco padrão para os testes
        mock_risk.max_daily_loss_pct = 0.02
        mock_risk.max_drawdown_pct = 0.05
        mock_risk.max_daily_profit_pct = 0.02
        mock_risk.cool_down_minutes = 5.0
        mock_risk.trading_start_time = "09:00:00"
        mock_risk.trading_end_time = "17:55:00"
        mock_risk.trade_type = "day_trade"
        
        yield

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
        risk.max_daily_profit_pct = 2.0  # 200% — desativa profit target para este teste

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

        om.close_positions("WIN")  # Nao falha sem MT5 ligado se for paper

    def test_get_net_position_paper_mode(self):
        """get_net_position deve retornar 0.0 se não estiver em modo 'live'."""
        om = OrderManager()
        with patch("src.execution.order_manager.execution_config") as mock_cfg:
            mock_cfg.mode = "paper"
            assert om.get_net_position("WIN") == 0.0

    def test_get_net_position_empty(self):
        """get_net_position deve retornar 0.0 se mt5.positions_get retornar None ou vazio."""
        om = OrderManager()
        with patch("src.execution.order_manager.execution_config") as mock_cfg:
            mock_cfg.mode = "live"
            with patch("src.execution.order_manager.mt5.positions_get", return_value=None):
                assert om.get_net_position("WIN") == 0.0
            with patch("src.execution.order_manager.mt5.positions_get", return_value=[]):
                assert om.get_net_position("WIN") == 0.0

    def test_get_net_position_calculates_correctly(self):
        """get_net_position deve somar volumes long/short que pertencem ao magic_number do sistema."""
        om = OrderManager()
        magic = om.magic_number

        # Cria mocks de posições MT5
        pos1 = MagicMock()
        pos1.magic = magic
        pos1.type = 0  # mt5.ORDER_TYPE_BUY
        pos1.volume = 2.0

        pos2 = MagicMock()
        pos2.magic = magic
        pos2.type = 1  # mt5.ORDER_TYPE_SELL
        pos2.volume = 1.0

        pos3 = MagicMock()
        pos3.magic = 999999  # Outro robô
        pos3.type = 0
        pos3.volume = 5.0

        # Como as constantes de MT5 mockado são mock objects, a comparação de tipo no código
        # será `pos.type == mt5.ORDER_TYPE_BUY`. Precisamos configurar os mocks das constantes MT5
        import MetaTrader5 as mt5
        pos1.type = mt5.ORDER_TYPE_BUY
        pos2.type = mt5.ORDER_TYPE_SELL
        pos3.type = mt5.ORDER_TYPE_BUY

        with patch("src.execution.order_manager.execution_config") as mock_cfg:
            mock_cfg.mode = "live"
            with patch("src.execution.order_manager.mt5.positions_get", return_value=[pos1, pos2, pos3]):
                # Long 2.0, Short 1.0 -> Net = 1.0
                assert om.get_net_position("WIN") == 1.0

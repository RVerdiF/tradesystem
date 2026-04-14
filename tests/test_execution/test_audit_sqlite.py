"""
Testes dedicados para AuditLogger (SQLite).

Verifica:
- Escrita correta de sinais, ordens e erros.
- Filtros de consulta (símbolo, intervalo de tempo, criticidade).
- Ordem de retorno (DESC por timestamp).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.execution.audit import AuditLogger
from src.db import get_connection, init_db


@pytest.fixture
def temp_audit(tmp_path):
    """Retorna um AuditLogger configurado para um banco SQLite temporário."""
    db_file = tmp_path / "audit_test.db"
    
    # Patch get_connection em src.db (afeta todo o sistema nesse contexto)
    # E também em src.execution.audit se já tiver sido importado
    with patch("src.db.DB_PATH", db_file):
        init_db(db_file)
        audit = AuditLogger()
        yield audit, db_file


def test_audit_signals_query_filtering(temp_audit):
    """Verifica filtros de símbolo e timestamp em query_signals."""
    audit, _ = temp_audit
    
    t0 = datetime.now() - timedelta(minutes=10)
    t1 = datetime.now()
    t2 = datetime.now() + timedelta(minutes=10)
    
    # Inserir sinais com diferentes timestamps e símbolos
    with patch("src.execution.audit.datetime") as mock_dt:
        mock_dt.now.return_value = t0
        audit.log_signal("PETR4", 1, 1, 0.5, 30.0)
        
        mock_dt.now.return_value = t1
        audit.log_signal("VALE3", -1, 1, 0.3, 80.0)
        
        mock_dt.now.return_value = t2
        audit.log_signal("PETR4", -1, 0, 0.0, 29.5)
        
    # Filtro por símbolo
    petr = audit.query_signals(symbol="PETR4")
    assert len(petr) == 2
    assert all(s["symbol"] == "PETR4" for s in petr)
    
    # Filtro por intervalo de tempo (t1 em diante)
    recent = audit.query_signals(start=t1.isoformat())
    assert len(recent) == 2
    assert recent[0]["symbol"] == "PETR4" # t2 (DESC)
    assert recent[1]["symbol"] == "VALE3" # t1
    
    # Filtro por intervalo (apenas t1)
    only_t1 = audit.query_signals(start=t1.isoformat(), end=t1.isoformat())
    assert len(only_t1) == 1
    assert only_t1[0]["symbol"] == "VALE3"


def test_audit_orders_query_filtering(temp_audit):
    """Verifica filtros de símbolo e timestamp em query_orders."""
    audit, _ = temp_audit
    
    t0 = datetime.now() - timedelta(minutes=10)
    t1 = datetime.now()
    t2 = datetime.now() + timedelta(minutes=10)

    with patch("src.execution.audit.datetime") as mock_dt:
        mock_dt.now.return_value = t0
        audit.log_order(101, "PETR4", "buy", 100, 30.0)

        mock_dt.now.return_value = t1
        audit.log_order(102, "VALE3", "sell", 200, 80.0)

        mock_dt.now.return_value = t2
        audit.log_order(103, "PETR4", "sell", 150, 31.0)
    
    res = audit.query_orders(symbol="VALE3")
    assert len(res) == 1
    assert res[0]["ticket"] == 102
    assert res[0]["volume"] == 200

    # Filtro por intervalo de tempo (t1 em diante)
    recent = audit.query_orders(start=t1.isoformat())
    assert len(recent) == 2
    assert recent[0]["ticket"] == 103 # t2 (DESC)
    assert recent[1]["ticket"] == 102 # t1

    # Filtro por intervalo (apenas t1)
    only_t1 = audit.query_orders(start=t1.isoformat(), end=t1.isoformat())
    assert len(only_t1) == 1
    assert only_t1[0]["ticket"] == 102


def test_audit_errors_query_filtering(temp_audit):
    """Verifica filtros de componente e critical_only em query_errors."""
    audit, _ = temp_audit
    
    audit.log_error("Engine", "Connection lost", critical=True)
    audit.log_error("Risk", "Exposure high", critical=False)
    audit.log_error("Engine", "Retry success", critical=False)
    
    # Filtro por componente
    engine_errors = audit.query_errors(component="Engine")
    assert len(engine_errors) == 2
    
    # Filtro por criticidade
    critical = audit.query_errors(critical_only=True)
    assert len(critical) == 1
    assert critical[0]["component"] == "Engine"
    
    # Ambos
    risk_critical = audit.query_errors(component="Risk", critical_only=True)
    assert len(risk_critical) == 0


def test_audit_empty_results(temp_audit):
    """Garante que retorna lista vazia em vez de erro se não houver registros."""
    audit, _ = temp_audit
    assert audit.query_signals() == []
    assert audit.query_orders() == []
    assert audit.query_errors() == []

def test_audit_log_order(temp_audit):
    """Verifica se log_order salva a ordem corretamente e se invoca o logger.success."""
    audit, _ = temp_audit

    with patch("src.execution.audit.logger") as mock_logger:
        audit.log_order(999, "WEGE3", "buy", 500, 35.5, comment="Test order")

        # Verify db insert
        res = audit.query_orders(symbol="WEGE3")
        assert len(res) == 1
        assert res[0]["ticket"] == 999
        assert res[0]["action"] == "buy"
        assert res[0]["volume"] == 500
        assert res[0]["price"] == 35.5
        assert res[0]["comment"] == "Test order"

        # Verify logger.success
        mock_logger.success.assert_called_once_with(
            "ORDEM ENVIADA: [{}] {} {} | Vol: {} | Preço: {}",
            999, "BUY", "WEGE3", 500, 35.5
        )

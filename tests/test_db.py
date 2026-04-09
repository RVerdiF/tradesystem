"""
Testes para src.db (Camada SQLite).

Verifica:
- get_connection (conexão válida, modo WAL)
- init_db (idempotência, criação de tabelas)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.db import get_connection, init_db, _ALL_DDL


@pytest.fixture
def temp_db_path(tmp_path):
    """Retorna um caminho para um banco temporário."""
    return tmp_path / "test_db.sqlite"


def test_get_connection_creates_dir_and_file(temp_db_path):
    """get_connection deve criar o diretório pai e o arquivo se não existirem."""
    assert not temp_db_path.exists()
    
    conn = get_connection(temp_db_path)
    conn.close()
    
    assert temp_db_path.exists()


def test_get_connection_wal_mode(temp_db_path):
    """A conexão deve estar em modo WAL para concorrência."""
    conn = get_connection(temp_db_path)
    res = conn.execute("PRAGMA journal_mode;").fetchone()
    conn.close()
    
    assert res["journal_mode"].upper() == "WAL"


def test_init_db_creates_all_tables(temp_db_path):
    """init_db deve criar as 4 tabelas esperadas."""
    init_db(temp_db_path)
    
    expected_tables = {
        "optimized_params",
        "audit_signals",
        "audit_orders",
        "audit_errors",
    }
    
    with get_connection(temp_db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        
        tables = {row["name"] for row in rows}
        
    # sqlite_sequence é criada pelo AUTOINCREMENT
    if "sqlite_sequence" in tables:
        tables.remove("sqlite_sequence")
        
    assert tables == expected_tables


def test_init_db_is_idempotent(temp_db_path):
    """Chamar init_db múltiplas vezes não deve causar erros."""
    init_db(temp_db_path)
    init_db(temp_db_path)  # Segunda vez
    
    with get_connection(temp_db_path) as conn:
        res = conn.execute("SELECT count(*) as count FROM sqlite_master WHERE type='table'").fetchone()
        # 4 tabelas + sqlite_sequence (devido ao autoincrement nas tabelas de auditoria)
        assert res["count"] >= 4


def test_foreign_keys_enabled(temp_db_path):
    """Foreign keys devem estar habilitadas por padrão."""
    conn = get_connection(temp_db_path)
    res = conn.execute("PRAGMA foreign_keys;").fetchone()
    conn.close()
    
    assert res["foreign_keys"] == 1

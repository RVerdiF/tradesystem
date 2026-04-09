"""
Camada de Persistência SQLite — TradeSystem5000.

Fornece:
- ``get_connection()`` — conexão WAL-mode thread-safe por chamada.
- ``init_db()``        — cria todas as tabelas (DDL idempotente).

Tabelas gerenciadas:
    optimized_params  — hiperparâmetros persistidos pelo params_store
    audit_signals     — sinais gerados pelo meta-modelo
    audit_orders      — ordens enviadas/simuladas
    audit_errors      — erros operacionais do sistema
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

# Importação via fallback para evitar import circular em testes com patch de DB_PATH
try:
    from config.settings import DB_PATH
except ImportError:
    DB_PATH = Path(__file__).resolve().parent.parent / "data" / "tradesystem.db"


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_CREATE_OPTIMIZED_PARAMS = """
CREATE TABLE IF NOT EXISTS optimized_params (
    symbol      TEXT    NOT NULL PRIMARY KEY,
    params      TEXT    NOT NULL,   -- JSON
    metadata    TEXT    NOT NULL,   -- JSON
    updated_at  TEXT    NOT NULL
);
"""

_CREATE_AUDIT_SIGNALS = """
CREATE TABLE IF NOT EXISTS audit_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    mode            TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    alpha_side      INTEGER NOT NULL,
    meta_label      INTEGER NOT NULL,
    kelly_fraction  REAL    NOT NULL,
    reference_price REAL    NOT NULL
);
"""

_CREATE_AUDIT_ORDERS = """
CREATE TABLE IF NOT EXISTS audit_orders (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    mode      TEXT    NOT NULL,
    ticket    INTEGER NOT NULL,
    symbol    TEXT    NOT NULL,
    action    TEXT    NOT NULL,
    volume    REAL    NOT NULL,
    price     REAL    NOT NULL,
    comment   TEXT    NOT NULL DEFAULT ''
);
"""

_CREATE_AUDIT_ERRORS = """
CREATE TABLE IF NOT EXISTS audit_errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    component   TEXT    NOT NULL,
    error_msg   TEXT    NOT NULL,
    critical    INTEGER NOT NULL DEFAULT 0  -- 0=False, 1=True
);
"""

_ALL_DDL = [
    _CREATE_OPTIMIZED_PARAMS,
    _CREATE_AUDIT_SIGNALS,
    _CREATE_AUDIT_ORDERS,
    _CREATE_AUDIT_ERRORS,
]


# ---------------------------------------------------------------------------
# Pública
# ---------------------------------------------------------------------------

def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """
    Abre e retorna uma nova conexão SQLite com WAL-mode habilitado.

    Cada chamador é responsável por fechar (ou usar via ``with`` statement).
    WAL permite leituras concorrentes sem bloquear escritas.

    Parameters
    ----------
    db_path : Path, optional
        Caminho para o banco. Default: ``DB_PATH`` de ``config.settings``.
    """
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    """
    Cria todas as tabelas caso ainda não existam (idempotente).

    Seguro chamar múltiplas vezes — usa ``CREATE TABLE IF NOT EXISTS``.
    """
    with get_connection(db_path) as conn:
        for ddl in _ALL_DDL:
            conn.execute(ddl)
        conn.commit()

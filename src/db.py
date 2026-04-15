"""Camada de Persistência SQLite — TradeSystem5000.

Fornece uma interface thread-safe para armazenamento de parâmetros otimizados,
sinais de auditoria, ordens e erros operacionais.

Funcionalidades:
- ``get_connection()`` — Conexão WAL-mode thread-safe por chamada.
- ``init_db()``        — Criação de tabelas via DDL idempotente.

Tabelas gerenciadas:
    - optimized_params: Hiperparâmetros persistidos pelo Tuner.
    - audit_signals: Sinais gerados pelo Meta-Modelo (lado do alpha, meta-label, kelly).
    - audit_orders: Registro de ordens enviadas ou simuladas.
    - audit_errors: Log de erros operacionais críticos ou informativos.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
"""

from __future__ import annotations

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
    """Abre e retorna uma nova conexão SQLite com WAL-mode habilitado.

    Cada chamador é responsável por fechar a conexão ou utilizá-la via contexto (with).
    O modo WAL (Write-Ahead Logging) permite leituras concorrentes sem bloquear escritas,
    essencial para a execução assíncrona do sistema.

    Parameters
    ----------
    db_path : Path, optional
        Caminho para o arquivo do banco de dados.
        Se None, utiliza o caminho definido em ``config.settings.DB_PATH``.

    Returns
    -------
    sqlite3.Connection
        Objeto de conexão configurado com ``row_factory = sqlite3.Row`` e WAL-mode.

    """
    if db_path is None:
        db_path = DB_PATH

    # Garante que o diretório existe
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row

    # Ativa WAL mode e Foreign Keys
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

    return conn


def init_db(db_path: Path | None = None) -> None:
    """Inicializa o esquema do banco de dados criando as tabelas necessárias.

    Esta função é idempotente; se as tabelas já existirem, nenhuma alteração será feita.
    Deve ser chamada no início da execução do sistema ou durante o setup.

    Parameters
    ----------
    db_path : Path, optional
        Caminho para o banco de dados. Default: ``DB_PATH``.

    Returns
    -------
    None

    """
    conn = get_connection(db_path)
    try:
        with conn:
            for ddl in _ALL_DDL:
                conn.execute(ddl)
    finally:
        conn.close()

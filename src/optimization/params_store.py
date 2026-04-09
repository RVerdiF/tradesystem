"""
Gerenciamento e persistência de hiperparâmetros otimizados.

Substitui a implementação anterior baseada em JSON flat-files por SQLite,
garantindo:
- Histórico auditável de otimizações por símbolo (updated_at).
- Consultas SQL ad-hoc via ``data/tradesystem.db``.
- API pública 100% compatível com a versão anterior.
"""

from __future__ import annotations

import json
from datetime import datetime

from loguru import logger

from src.db import get_connection, init_db

# Garante que as tabelas existam na primeira importação
init_db()


def _normalize_symbol(symbol: str) -> str:
    """Remove sufixo .SA para manter coerência entre MT5 e Yahoo Finance."""
    return symbol.replace(".SA", "")


def save_optimized_params(symbol: str, params: dict, metadata: dict | None = None) -> None:
    """
    Persiste os parâmetros otimizados no banco SQLite.

    Usa ``INSERT OR REPLACE`` — sobrescreve se o símbolo já existir.

    Parameters
    ----------
    symbol : str
        Ativo correspondente (ex: PETR4 ou PETR4.SA).
    params : dict
        Hiperparâmetros otimizados.
    metadata : dict | None
        Dados extras do Optuna (Sharpe, DSR, data, n_trials).
    """
    symbol = _normalize_symbol(symbol)

    if metadata is None:
        metadata = {}

    metadata["timestamp"] = datetime.now().isoformat()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO optimized_params (symbol, params, metadata, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (symbol, json.dumps(params), json.dumps(metadata), metadata["timestamp"]),
        )
        conn.commit()

    logger.success(f"Parâmetros otimizados de {symbol} salvos no banco SQLite.")


def load_optimized_params(symbol: str) -> dict | None:
    """
    Lê os parâmetros do banco SQLite.

    Returns
    -------
    dict com chaves ``symbol``, ``params`` e ``metadata``, ou ``None``
    se o símbolo não tiver sido otimizado ainda.
    """
    symbol = _normalize_symbol(symbol)

    with get_connection() as conn:
        row = conn.execute(
            "SELECT symbol, params, metadata FROM optimized_params WHERE symbol = ?",
            (symbol,),
        ).fetchone()

    if row is None:
        logger.info(f"Nenhum parâmetro encontrado no banco para {symbol}.")
        return None

    data = {
        "symbol": row["symbol"],
        "params": json.loads(row["params"]),
        "metadata": json.loads(row["metadata"]),
    }

    ts = data["metadata"].get("timestamp", "N/A")
    logger.info(f"Parâmetros de {symbol} carregados (Otimizados em: {ts}).")
    return data


def params_exist(symbol: str) -> bool:
    """Verifica se existem parâmetros persistidos para o símbolo."""
    symbol = _normalize_symbol(symbol)

    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM optimized_params WHERE symbol = ?",
            (symbol,),
        ).fetchone()

    return row is not None

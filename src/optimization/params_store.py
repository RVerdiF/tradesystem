"""Gerenciamento e Persistência de Hiperparâmetros — TradeSystem5000.

Este módulo implementa a camada de armazenamento para os parâmetros otimizados
pelo Tuner, utilizando o banco de dados SQLite central para auditoria e
recuperação eficiente.

Funcionalidades:
- **save_optimized_params**: Persistência de dicionários de parâmetros e metadados.
- **load_optimized_params**: Recuperação com suporte a fallback de símbolos contínuos.
- **params_exist**: Verificação rápida de presença de configurações para um ativo.
- Normalização de símbolos para consistência entre MT5 e Yahoo Finance.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from loguru import logger

from src.db import get_connection, init_db

# Garante que as tabelas existam na primeira importação
init_db()


def _normalize_symbol(symbol: str) -> str:
    """Remove sufixo .SA para manter coerência entre MT5 e Yahoo Finance."""
    return symbol.replace(".SA", "")


def get_continuous_symbol(symbol: str) -> str:
    """Retorna o símbolo contínuo para ativos da B3 que possuem séries mensais.

    Ex: WINJ26 -> WIN$, WDOM25 -> WDO$.

    Parameters
    ----------
    symbol : str
        Ativo de entrada B3 original para serialização.

    Returns
    -------
    str
        Ticker consolidado de negociação contínua.

    """
    # Regex para WIN, WDO, IND, DOL + Letra do Mês + Ano (2 dígitos)
    match = re.match(r"^(WIN|WDO|IND|DOL)[FGHJKMNQUVXZ]\d{2}$", symbol, re.IGNORECASE)
    if match:
        return f"{match.group(1).upper()}$"
    return symbol


def save_optimized_params(symbol: str, params: dict, metadata: dict | None = None) -> None:
    """Salva e armazena os parâmetros otimizados no banco SQLite.

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
    """Lê os parâmetros do banco SQLite.

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
        # Fallback para símbolo contínuo (ex: WINJ26 -> WIN$)
        continuous_symbol = get_continuous_symbol(symbol)
        if continuous_symbol != symbol:
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT symbol, params, metadata FROM optimized_params WHERE symbol = ?",
                    (continuous_symbol,),
                ).fetchone()
            if row:
                logger.info(f"Usando fallback contínuo: {continuous_symbol} para {symbol}")

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

    if row is not None:
        return True

    # Fallback
    continuous_symbol = get_continuous_symbol(symbol)
    if continuous_symbol != symbol:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM optimized_params WHERE symbol = ?",
                (continuous_symbol,),
            ).fetchone()
        return row is not None

    return False

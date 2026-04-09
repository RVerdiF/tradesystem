"""
6.4 — Auditoria e Logging.

Registra todas as ações do sistema em um banco SQLite estruturado,
facilmente consultável para reconciliação e auditoria posterior.
Também exibe alertas visuais via Loguru.

Tabelas utilizadas (criadas por ``src.db.init_db``):
    audit_signals  — sinais gerados pelo meta-modelo
    audit_orders   — ordens enviadas ou simuladas
    audit_errors   — erros operacionais do sistema
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from config.settings import execution_config
from src.db import get_connection, init_db

# Garante que as tabelas existam na primeira importação
init_db()


class AuditLogger:
    """
    Sistema de auditoria para persistir logs operacionais em SQLite.

    A API pública (``log_signal``, ``log_order``, ``log_error``) é idêntica
    à versão anterior baseada em JSONL, garantindo zero impacto nos chamadores.
    """

    def __init__(self) -> None:
        # Sem estado de arquivo — conexões são abertas por chamada
        pass

    # ------------------------------------------------------------------
    # Escrita
    # ------------------------------------------------------------------

    def log_signal(
        self,
        symbol: str,
        alpha_side: int,
        meta_label: int,
        kelly_fraction: float,
        price: float,
    ) -> None:
        """Audita a geração de um sinal do modelo."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_signals
                  (timestamp, mode, symbol, alpha_side, meta_label, kelly_fraction, reference_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    execution_config.mode,
                    symbol,
                    alpha_side,
                    meta_label,
                    kelly_fraction,
                    price,
                ),
            )
            conn.commit()

        # Log visual rápido (Loguru — terminal apenas)
        if meta_label == 1 and kelly_fraction > 0:
            logger.info(
                "SINAL APROVADO: {} | Side: {} | P: {} | f*: {:.2f}",
                symbol, alpha_side, price, kelly_fraction,
            )
        elif alpha_side != 0:
            logger.debug(
                "SINAL REJEITADO (Meta=0 ou f=0): {} | Side: {}", symbol, alpha_side
            )

    def log_order(
        self,
        ticket: int,
        symbol: str,
        action: str,
        volume: float,
        price: float,
        comment: str = "",
    ) -> None:
        """Audita o envio ou fechamento de uma ordem na corretora."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_orders
                  (timestamp, mode, ticket, symbol, action, volume, price, comment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    execution_config.mode,
                    ticket,
                    symbol,
                    action,
                    volume,
                    price,
                    comment,
                ),
            )
            conn.commit()

        logger.success(
            "ORDEM ENVIADA: [{}] {} {} | Vol: {} | Preço: {}",
            ticket, action.upper(), symbol, volume, price,
        )

    def log_error(
        self, component: str, error_msg: str, critical: bool = False
    ) -> None:
        """Audita erros operacionais do sistema."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_errors (timestamp, component, error_msg, critical)
                VALUES (?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    component,
                    error_msg,
                    int(critical),
                ),
            )
            conn.commit()

        if critical:
            logger.critical("CRITICAL ERROR ({}): {}", component, error_msg)
        else:
            logger.error("ERROR ({}): {}", component, error_msg)

    # ------------------------------------------------------------------
    # Consulta (reconciliação / debugging)
    # ------------------------------------------------------------------

    def query_signals(
        self,
        symbol: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retorna sinais registrados, com filtros opcionais.

        Parameters
        ----------
        symbol : str, optional
            Filtra por ativo.
        start : str, optional
            ISO timestamp mínimo (inclusive).
        end : str, optional
            ISO timestamp máximo (inclusive).
        """
        query = "SELECT * FROM audit_signals WHERE 1=1"
        params: list[Any] = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)
        query += " ORDER BY timestamp DESC"

        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(r) for r in rows]

    def query_orders(
        self,
        symbol: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retorna ordens registradas, com filtros opcionais.

        Parameters
        ----------
        symbol : str, optional
            Filtra por ativo.
        start : str, optional
            ISO timestamp mínimo (inclusive).
        end : str, optional
            ISO timestamp máximo (inclusive).
        """
        query = "SELECT * FROM audit_orders WHERE 1=1"
        params: list[Any] = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)
        query += " ORDER BY timestamp DESC"

        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(r) for r in rows]

    def query_errors(
        self,
        component: str | None = None,
        critical_only: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Retorna erros registrados, com filtros opcionais.

        Parameters
        ----------
        component : str, optional
            Filtra por componente (ex: 'RiskManager').
        critical_only : bool
            Se True, retorna apenas erros críticos.
        """
        query = "SELECT * FROM audit_errors WHERE 1=1"
        params: list[Any] = []
        if component:
            query += " AND component = ?"
            params.append(component)
        if critical_only:
            query += " AND critical = 1"
        query += " ORDER BY timestamp DESC"

        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(r) for r in rows]


# Instância global (singleton-like)
audit = AuditLogger()

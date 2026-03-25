"""
6.4 — Auditoria e Logging.

Registra todas as ações do sistema em arquivos de log estruturados (JSON/CSV)
facilmente parseáveis para reconciliação e auditoria posterior.
Também alerta via Loguru.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import execution_config


class AuditLogger:
    """
    Sistema de auditoria para salvar logs operacionais estruturados.
    """

    def __init__(self, log_dir: str | Path = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Arquivos de log específicos
        self.trade_log_file = self.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.signal_log_file = self.log_dir / f"signals_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.error_log_file = self.log_dir / "critical_errors.log"

    def _write_jsonl(self, filepath: Path, data: dict[str, Any]) -> None:
        """Escreve um registro em formato JSON Lines."""
        data["timestamp"] = datetime.now().isoformat()
        data["mode"] = execution_config.mode
        
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    def log_signal(
        self, 
        symbol: str, 
        alpha_side: int, 
        meta_label: int, 
        kelly_fraction: float, 
        price: float
    ) -> None:
        """Audita a geração de um sinal do modelo."""
        data = {
            "event": "SIGNAL_GENERATED",
            "symbol": symbol,
            "alpha_side": alpha_side,
            "meta_label": meta_label,
            "kelly_fraction": kelly_fraction,
            "reference_price": price
        }
        self._write_jsonl(self.signal_log_file, data)
        # Log visual rápido
        if meta_label == 1 and kelly_fraction > 0:
            logger.info("SINAL APROVADO: {} | Side: {} | P: {} | f*: {:.2f}", symbol, alpha_side, price, kelly_fraction)
        elif alpha_side != 0:
            logger.debug("SINAL REJEITADO (Meta=0 ou f=0): {} | Side: {}", symbol, alpha_side)

    def log_order(
        self, 
        ticket: int, 
        symbol: str, 
        action: str, 
        volume: float, 
        price: float, 
        comment: str = ""
    ) -> None:
        """Audita o envio ou fechamento de uma ordem na corretora."""
        data = {
            "event": "ORDER_EXECUTED",
            "ticket": ticket,
            "symbol": symbol,
            "action": action,         # 'buy', 'sell', 'close'
            "volume": volume,
            "price": price,
            "comment": comment
        }
        self._write_jsonl(self.trade_log_file, data)
        logger.success("ORDEM ENVIADA: [{}] {} {} | Vol: {} | Preço: {}", ticket, action.upper(), symbol, volume, price)

    def log_error(self, component: str, error_msg: str, critical: bool = False) -> None:
        """Audita erros operacionais do sistema."""
        msg = f"[{datetime.now().isoformat()}] [{component}] {error_msg}\n"
        with open(self.error_log_file, "a", encoding="utf-8") as f:
            f.write(msg)
            
        if critical:
            logger.critical("CRITICAL ERROR ({}): {}", component, error_msg)
        else:
            logger.error("ERROR ({}): {}", component, error_msg)


# Instância global (singleton-like)
audit = AuditLogger()

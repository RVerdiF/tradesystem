"""
6.2 — Gerenciador de Ordens (MT5 Integration).

Lida com a comunicação de baixo nível com a MetaTrader 5:
- order_send com validação e slicing
- tracking de posições em aberto
- fechamento de posições
"""

from __future__ import annotations

import MetaTrader5 as mt5
from loguru import logger

from config.settings import execution_config, mt5_config
from src.execution.audit import audit


class OrderManager:
    """
    Controlador de ordens para MetaTrader 5.
    """

    def __init__(self) -> None:
        self.magic_number = execution_config.magic_number
        self.deviation = execution_config.max_slippage_ticks

    def wait_order_result(self, result: tuple | None) -> bool:
        """Avalia o retcode do Metatrader."""
        if result is None:
            audit.log_error("OrderManager", "Retorno de order_send foi None. Falha de comunicação.", critical=True)
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = f"Ordem rejeitada. Retcode: {result.retcode} - {result.comment}"
            audit.log_error("OrderManager", msg)
            return False

        # Sucesso
        audit.log_order(
            ticket=result.order,
            symbol=result.request.symbol,
            action="Executed",
            volume=result.volume,
            price=result.price,
            comment=result.comment
        )
        return True

    def send_market_order(self, symbol: str, action: str, volume: float) -> bool:
        """
        Envia uma ordem a mercado.
        
        Parameters
        ----------
        symbol : str
            Ativo (ex: 'WINZ25')
        action : str
            'buy' ou 'sell'
        volume : float
            Quantidade (lotes)
            
        Returns
        -------
        bool
            True se executada com sucesso.
        """
        # Proteção: só no modo live tenta enviar ordens reais
        if execution_config.mode != "live":
            logger.info("[PAPER] Simulação: {} {} lotes de {}", action.upper(), volume, symbol)
            # Log simulado
            audit.log_order(ticket=-1, symbol=symbol, action=action.upper() + "_SIMULADA", volume=volume, price=0.0)
            return True

        point = mt5.symbol_info(symbol).point
        price = mt5.symbol_info_tick(symbol).ask if action == 'buy' else mt5.symbol_info_tick(symbol).bid
        type_mt5 = mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": type_mt5,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": "TS5000",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN, # ou FOK dependendo da corretora B3
        }

        # Envio de ordem real
        logger.debug("Enviando requisição: {}", request)
        result = mt5.order_send(request)
        
        return self.wait_order_result(result)

    def close_positions(self, symbol: str) -> None:
        """Fecha todas as posições em aberto do sistema para um ativo."""
        if execution_config.mode != "live":
            logger.info("[PAPER] Simulação: CLOSE POSITIONS para {}", symbol)
            audit.log_order(ticket=-1, symbol=symbol, action="CLOSE_ALL_SIMULADA", volume=0.0, price=0.0)
            return

        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return

        for pos in positions:
            # Só fecha posições originadas pelo sistema
            if pos.magic == self.magic_number:
                action_close = "sell" if pos.type == mt5.ORDER_TYPE_BUY else "buy"
                logger.info("Fechando Posição {} de {}", pos.ticket, symbol)
                self.send_market_order(symbol, action=action_close, volume=pos.volume)
                
    def get_net_position(self, symbol: str) -> float:
        """Retorna a exposição líquida aberta (positiva = long, negativa = short)."""
        if execution_config.mode != "live":
            return 0.0
            
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return 0.0
            
        net_vol = 0.0
        for pos in positions:
            if pos.magic == self.magic_number:
                vol = pos.volume if pos.type == mt5.ORDER_TYPE_BUY else -pos.volume
                net_vol += vol
                
        return net_vol

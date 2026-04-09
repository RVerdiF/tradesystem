"""
6.3 — Circuit Breakers (Gerenciamento de Risco da Conta).

Impede operações caso os limites de Drawdown, Exposição Máxima 
ou Perda Diária Máxima sejam atingidos.
"""

from __future__ import annotations

import datetime

from loguru import logger

from config.settings import risk_config, execution_config
from src.execution.audit import audit


class RiskManager:
    """
    Gerenciador de Risco Macro (Conta/Global).
    Avalia se o sistema como um todo está autorizado a enviar novas ordens.
    """

    def __init__(
        self, 
        start_balance: float | None = None,
        trade_type: str = risk_config.trade_type,
        start_time: str = risk_config.trading_start_time,
        end_time: str = risk_config.trading_end_time
    ) -> None:
        """
        Parameters
        ----------
        start_balance : float, opcional
            Saldo inicial para cálculo de perda diária. 
            No MT5 Real, isso é carregado via `account_info`.
        trade_type : str
            'day_trade' ou 'swing_trade'.
        start_time : str
            Horário de início (HH:MM:SS).
        end_time : str
            Horário de término (HH:MM:SS).
        """
        self.start_balance = start_balance
        self.current_equity = start_balance
        self.highest_equity = start_balance
        
        self.max_daily_loss_pct = risk_config.max_daily_loss_pct
        self.max_drawdown_pct = risk_config.max_drawdown_pct
        
        self.trade_type = trade_type
        self.start_time = datetime.time.fromisoformat(start_time)
        self.end_time = datetime.time.fromisoformat(end_time)
        
        self.is_halted = False
        self.halt_reason = ""
        
        # Reset diário
        self.last_trading_day = datetime.date.today()

    def update_equity(self, balance: float, equity: float) -> None:
        """
        Atualiza o estado da conta (chamado a cada ciclo/tick).
        
        No MT5 balance = saldo fechado, equity = saldo + lucro flutuante.
        """
        today = datetime.date.today()
        
        # Se mudou o dia de trading, reseta o balanço inicial para o dia
        if today > self.last_trading_day:
            self.start_balance = balance  # Novo dia começa com o saldo atual
            self.last_trading_day = today
            self.is_halted = False
            self.halt_reason = ""
            logger.info("Novo dia de trading. Saldo inicial resetado para: {:.2f}", self.start_balance)
            
        if self.start_balance is None:
            self.start_balance = balance
            self.highest_equity = equity

        self.current_equity = equity
        
        if equity > self.highest_equity:
            self.highest_equity = equity

        self._check_circuit_breakers()

    def _check_circuit_breakers(self) -> None:
        """
        Verifica se alguma regra de risco macro foi violada.
        """
        # Se estiver travado por PnL ou Drawdown, não libera por horário
        if self.is_halted and "WINDOW" not in self.halt_reason:
            return

        # 0. Verificação de Horário
        now = datetime.datetime.now().time()
        if now < self.start_time or now > self.end_time:
            self.is_halted = True
            self.halt_reason = f"OUTSIDE TRADING WINDOW ({now.strftime('%H:%M:%S')})"
            return
        elif "WINDOW" in self.halt_reason:
            # Se estava travado por horário e agora está dentro, libera
            self.is_halted = False
            self.halt_reason = ""

        # 1. Perda Diária (Daily Loss)
        daily_pnl_pct = (self.current_equity / self.start_balance) - 1.0
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            self.is_halted = True
            self.halt_reason = f"MAX DAILY LOSS REACHED ({daily_pnl_pct:.2%})"
            audit.log_error("RiskManager", self.halt_reason, critical=True)
            return

        # 2. Maximum Drawdown (Conta Global)
        drawdown_pct = (self.current_equity / self.highest_equity) - 1.0
        if drawdown_pct <= -self.max_drawdown_pct:
            self.is_halted = True
            self.halt_reason = f"MAX DRAWDOWN REACHED ({drawdown_pct:.2%})"
            audit.log_error("RiskManager", self.halt_reason, critical=True)
            return

    def can_trade(self) -> bool:
        """
        Retorna True se o sistema estiver livre para enviar ordens.
        """
        if self.is_halted:
            logger.warning("TRADING HALTED: {}", self.halt_reason)
            return False
            
        return True

    def validate_order(self, current_exposure: float, new_volume: float, max_exposure: float) -> bool:
        """
        Verifica exposição máxima por ativo/conta antes de enviar uma ordem.
        """
        if self.is_halted:
            return False
            
        if (current_exposure + new_volume) > max_exposure:
            msg = f"Rejeitado: Exposição {current_exposure + new_volume} excede limite {max_exposure}."
            logger.warning(msg)
            audit.log_error("RiskManager", msg)
            return False
            
        return True

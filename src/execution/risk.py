"""Gerenciamento de Risco e Circuit Breakers — TradeSystem5000.

Este módulo implementa a camada de proteção macro (nível de conta), impedindo
operações se limites críticos de perda, lucro ou tempo forem atingidos.

Regras de Proteção:
- **Daily Loss**: Limite de perda percentual diária sobre o saldo inicial.
- **Daily Profit**: Limite de lucro percentual diário (protege ganhos).
- **Max Drawdown**: Limite de queda a partir do pico de equity da conta.
- **Trading Window**: Restrição horária para operações (Day Trade safety).
- **Cool-Down**: Período de resfriamento obrigatório após uma saída de posição.
- **Exposure Limits**: Validação de volume máximo por ativo.

Estados do Sistema
------------------
- ACTIVE          : Sistema livre para operar.
- HALTED_FOR_DAY  : Halt permanente para o dia (loss, profit ou drawdown).
- OUTSIDE_WINDOW  : Fora do horário de operação. Limpa automaticamente.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
"""

from __future__ import annotations

import datetime

from loguru import logger

from config.settings import risk_config
from src.execution.audit import audit

# Constantes de estado (evita typos em comparações de string)
STATE_ACTIVE = "ACTIVE"
STATE_HALTED_FOR_DAY = "HALTED_FOR_DAY"
STATE_OUTSIDE_WINDOW = "OUTSIDE_WINDOW"


class RiskManager:
    """Gerenciador de Risco Macro (Conta/Global).

    Avalia se o sistema como um todo está autorizado a enviar novas ordens.
    """

    def __init__(
        self,
        start_balance: float | None = None,
        trade_type: str = risk_config.trade_type,
        start_time: str = risk_config.trading_start_time,
        end_time: str = risk_config.trading_end_time,
    ) -> None:
        """Inicializa o RiskManager.

        Parameters
        ----------
        start_balance : float, opcional
            Saldo inicial para cálculo de perda/lucro diário.
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
        self.max_daily_profit_pct = risk_config.max_daily_profit_pct
        self.cool_down_minutes = risk_config.cool_down_minutes

        self.trade_type = trade_type
        self.start_time = datetime.time.fromisoformat(start_time)
        self.end_time = datetime.time.fromisoformat(end_time)

        # Estado canónico do sistema
        self.system_state: str = STATE_ACTIVE

        # Atributos de compatibilidade retroativa — sempre mantidos em sincronia
        # com system_state via _set_state(). NÃO altere diretamente em produção.
        self.is_halted: bool = False
        self.halt_reason: str = ""

        # Temporizador do cool-down por ativo
        self._cool_down_until: dict[str, datetime.datetime] = {}

        # Reset diário
        self.last_trading_day = datetime.date.today()

    # ------------------------------------------------------------------
    # API Pública
    # ------------------------------------------------------------------

    def update_equity(self, balance: float, equity: float) -> None:
        """Atualiza o estado da conta (chamado a cada ciclo/tick).

        No MT5: balance = saldo fechado, equity = saldo + lucro flutuante.
        """
        today = datetime.date.today()

        # Se mudou o dia de trading, reseta para o novo dia
        if today > self.last_trading_day:
            self.start_balance = balance
            self.last_trading_day = today
            self._cool_down_until.clear()  # Descarta cool-down do dia anterior
            self._set_state(STATE_ACTIVE, "")
            logger.info(
                "Novo dia de trading. Saldo inicial resetado para: {:.2f}", self.start_balance
            )

        if self.start_balance is None:
            self.start_balance = balance
            self.highest_equity = equity

        self.current_equity = equity

        if equity > self.highest_equity:
            self.highest_equity = equity

        self._check_circuit_breakers()

    def notify_trade_closed(self, symbol: str = "GLOBAL") -> None:
        """Inicia o período de cool-down após uma saída de posição para flat.

        Deve ser chamado APENAS na saída por circuit breaker ou TP/SL (não em stop-and-reverse),
        para evitar bloquear a gestão de posições recém-abertas.
        Não-operacional se cool_down_minutes <= 0 ou se o sistema já estiver
        em HALTED_FOR_DAY.
        """
        if self.cool_down_minutes <= 0:
            return
        if self.system_state == STATE_HALTED_FOR_DAY:
            return
        if self.system_state == STATE_OUTSIDE_WINDOW:
            return  # Window-based closes don't warrant a cool-down; daily reset handles cleanup

        cool_down_until = datetime.datetime.now() + datetime.timedelta(
            minutes=self.cool_down_minutes
        )
        self._cool_down_until[symbol] = cool_down_until

        reason = f"COOL_DOWN (until {cool_down_until.strftime('%H:%M:%S')} for {symbol})"
        audit.log_error(
            "RiskManager", reason, critical=False
        )  # Persiste no SQLite para análise pós-trade
        logger.info(
            "Cool-down ativado para {} até: {}", symbol, cool_down_until.strftime("%H:%M:%S")
        )

    def can_trade(self, symbol: str = "GLOBAL") -> bool:
        """Retorna True se o sistema e o ativo estiverem livres para enviar ordens."""
        if self.is_halted:
            logger.warning("TRADING HALTED: {}", self.halt_reason)
            return False

        if symbol in self._cool_down_until:
            if datetime.datetime.now() >= self._cool_down_until[symbol]:
                del self._cool_down_until[symbol]
                logger.info(
                    "Cool-down expirado para {}. Sistema reativado para este ativo.", symbol
                )
            else:
                logger.warning(
                    "TRADING BLOCKED FOR {}: COOL_DOWN until {}",
                    symbol,
                    self._cool_down_until[symbol].strftime("%H:%M:%S"),
                )
                return False

        return True

    def validate_order(
        self, current_exposure: float, new_volume: float, max_exposure: float
    ) -> bool:
        """Verifica exposição máxima por ativo/conta antes de enviar uma ordem."""
        if self.is_halted:
            return False

        if (current_exposure + new_volume) > max_exposure:
            msg = f"Rejeitado: Exposição {current_exposure + new_volume} excede limite {max_exposure}."
            logger.warning(msg)
            audit.log_error("RiskManager", msg)
            return False

        return True

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _set_state(self, state: str, reason: str) -> None:
        """Transição de estado atômica.

        Mantém system_state, is_halted e halt_reason em sincronia.
        Único ponto de mutação de estado.
        """
        self.system_state = state
        self.is_halted = state != STATE_ACTIVE
        self.halt_reason = reason

    def _check_circuit_breakers(self) -> None:
        """Avalia todas as regras de risco macro.

        Ordem de prioridade:
        1. Halt permanente do dia (só o reset diário pode limpar)
        2. Janela de horário
        3. Daily Loss
        4. Daily Profit
        5. Max Drawdown.
        """
        # --- 1. Halt permanente do dia (PnL ou Drawdown) ---
        if self.system_state == STATE_HALTED_FOR_DAY:
            return

        # --- 2. Janela de horário ---
        now = datetime.datetime.now().time()
        if now < self.start_time or now > self.end_time:
            self._set_state(
                STATE_OUTSIDE_WINDOW, f"OUTSIDE TRADING WINDOW ({now.strftime('%H:%M:%S')})"
            )
            return
        elif self.system_state == STATE_OUTSIDE_WINDOW:
            # Estava fora do horário, agora está dentro — reativa
            self._set_state(STATE_ACTIVE, "")

        # --- 3. Perda Diária (Daily Loss) ---
        daily_pnl_pct = (self.current_equity / self.start_balance) - 1.0
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            reason = f"MAX DAILY LOSS REACHED ({daily_pnl_pct:.2%})"
            self._set_state(STATE_HALTED_FOR_DAY, reason)
            audit.log_error("RiskManager", reason, critical=True)
            return

        # --- 4. Meta de Lucro Diário (Daily Profit Target) ---
        if daily_pnl_pct >= self.max_daily_profit_pct:
            reason = f"MAX DAILY PROFIT REACHED ({daily_pnl_pct:.2%})"
            self._set_state(STATE_HALTED_FOR_DAY, reason)
            audit.log_error("RiskManager", reason, critical=True)
            return

        # --- 5. Maximum Drawdown (Conta Global) ---
        drawdown_pct = (self.current_equity / self.highest_equity) - 1.0
        if drawdown_pct <= -self.max_drawdown_pct:
            reason = f"MAX DRAWDOWN REACHED ({drawdown_pct:.2%})"
            self._set_state(STATE_HALTED_FOR_DAY, reason)
            audit.log_error("RiskManager", reason, critical=True)
            return

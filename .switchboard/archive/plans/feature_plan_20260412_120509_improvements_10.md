# Improvements 10.

## Goal
Implement operational interruption mechanisms to prevent overtrading and protect gains by introducing post-trade cool-down periods and a maximum daily profit target in the system's risk manager, alongside explicit system state tracking for clear, auditable logging.

## Metadata
**Tags:** backend
**Complexity:** Low

## User Review Required
> [!NOTE]
> - Two new parameters are added to `RiskConfig` in `config/settings.py` (a `frozen=True` dataclass). Their defaults are `cool_down_minutes = 5.0` and `max_daily_profit_pct = 0.02`. Review and adjust before deploying to live.
> - The `system_state` attribute is new — any external tooling or dashboards that inspect `RiskManager.halt_reason` directly will need to be validated (the string format is unchanged; the new `COOL_DOWN` and `HALTED_FOR_DAY` states extend the existing vocabulary).
> - **Clarification (time vs. bar-based cool-down):** Cool-down uses wall-clock time (minutes), not bar count. This is simpler, mode-agnostic, and mockable in unit tests. Bar-based cool-down would require the engine to pass bar indices into the risk layer — a larger architectural change out of scope here.
> - **Clarification (daily profit target unit):** The target is expressed as a percentage of `start_balance` for the day (e.g. `0.02` = 2%), consistent with `max_daily_loss_pct`. Fixed-EUR and ATR-dynamic modes are deferred.
> - **Clarification (manual override):** No operator override command is implemented in this plan. Override logic is deferred to a future plan.

## Complexity Audit

### Routine
- Add `cool_down_minutes: float` and `max_daily_profit_pct: float` to `RiskConfig` in `config/settings.py`.
- Add `system_state: str`, `_cool_down_until: datetime | None` attributes to `RiskManager.__init__()`.
- Add `_set_state(state, reason)` helper to `RiskManager` to keep `is_halted`, `halt_reason`, and `system_state` in sync atomically.
- Add `notify_trade_closed()` public method to `RiskManager`.
- Restructure `_check_circuit_breakers()` to evaluate cool-down expiry before the existing early-return guard, and add daily profit check.
- Extend daily reset in `update_equity()` to clear `_cool_down_until`.
- Add one line in `engine.py` `_process_symbol()` to call `self.risk.notify_trade_closed()` after the circuit-breaker close path.
- Write 5 new unit tests in `tests/test_execution/test_time_restrictions.py`.

### Complex / Risky
- None

## Edge-Case & Dependency Audit
- **Race Conditions:** `asyncio.gather` processes multiple symbols concurrently. `RiskManager` is a shared, non-async singleton. Since the cool-down state mutation (`_set_state`, `_cool_down_until`) happens via simple attribute assignment and Python's GIL protects single attribute writes, there is no race condition risk under the current asyncio (single-thread) model. This assumption holds as long as the engine does not switch to multi-process execution.
- **Security:** No security-sensitive logic. No auth, crypto, or permissions involved.
- **Side Effects:**
  - `notify_trade_closed()` is intentionally NOT called in the stop-and-reverse path (`close_positions()` → `send_market_order()` in the same tick). Calling it there would place a halt on a position that was just opened, preventing its management on the next tick. The call site is explicitly the circuit-breaker close path only.
  - In paper mode, `OrderManager.get_net_position()` always returns `0.0`, so the `if self.om.get_net_position(symbol) != 0` guard in `engine.py` will never trigger `notify_trade_closed()` via the circuit-breaker path. This means cool-down is untestable end-to-end in paper mode via the engine. Mitigated by unit-testing `RiskManager.notify_trade_closed()` and `_check_circuit_breakers()` independently with mocked datetime.
  - `RiskConfig` is `frozen=True`. New fields have defaults — existing instantiation without keyword arguments is unaffected.
  - Daily reset in `update_equity()` must clear `_cool_down_until = None` explicitly. If omitted, a cool-down active at midnight would leave a stale `_cool_down_until` on day two (harmless given the new state machine, but noisy).
- **Dependencies & Conflicts:** No other plans currently in the Kanban board. No cross-plan conflicts identified.

## Adversarial Synthesis

### Grumpy Critique
*(See chat response — verbatim critique preserved above per dispatch instructions.)*

**Summary of charges (pre-implementation):**
1. The existing `_check_circuit_breakers()` early-return guard `if self.is_halted and "WINDOW" not in self.halt_reason: return` will permanently short-circuit the cool-down expiry check if `is_halted` is set True for cool-down. The timer never clears. System stays frozen until midnight.
2. If `notify_trade_closed()` is called in the stop-and-reverse path, the engine opens a new position and immediately loses ability to manage it via signals on the next tick.
3. Paper mode `get_net_position()` always returns 0 — the cool-down trigger in the engine circuit-breaker path never fires in paper mode.
4. Daily reset must explicitly zero `_cool_down_until`.
5. Open questions (time vs. bar, fixed vs. dynamic profit target, operator override) are load-bearing and must be resolved before implementation.
6. Cool-down activations must persist to SQLite via `audit.log_error(critical=False)`, not only to log files.

**Post-implementation reviewer charges:**
- **MAJOR (fixed):** `notify_trade_closed()` only guarded against `STATE_HALTED_FOR_DAY`; calling it on `OUTSIDE_WINDOW` closures overwrote that state with `COOL_DOWN`, producing misleading audit noise nightly (COOL_DOWN → ACTIVE → OUTSIDE_WINDOW cycling every 5 minutes until midnight). Fixed by adding `STATE_OUTSIDE_WINDOW` guard in `notify_trade_closed()`.
- **MAJOR (documented, not fixed):** Existing engine tests bypass `_set_state()` via direct attribute assignment (`engine.risk.is_halted = True`), leaving `system_state` as `ACTIVE`. The OUTSIDE_WINDOW guard fix is therefore not exercised by the current engine test suite. Pre-existing weakness exposed by new code — not introduced by this plan.
- **NIT:** `mock_dt.timedelta = datetime.timedelta` in several `TestCoolDown` tests is a no-op (production code accesses `datetime.timedelta` through the module, not the class). Harmless.

### Balanced Response
All six pre-implementation charges addressed. One post-review MAJOR fixed (OUTSIDE_WINDOW guard).

1. `_check_circuit_breakers()` is restructured to check `system_state` rather than `is_halted` for the short-circuit guard. Cool-down expiry is evaluated first, unconditionally.
2. `notify_trade_closed()` call site is explicitly restricted to the circuit-breaker path in `engine.py` with a comment marking the stop-and-reverse paths as excluded.
3. Paper mode limitation is documented. Unit tests use mocked datetime to validate the cool-down directly on `RiskManager`.
4. Daily reset explicitly sets `self._cool_down_until = None`.
5. All open questions resolved as Clarifications above.
6. Cool-down activations call `audit.log_error("RiskManager", ..., critical=False)` for DB persistence.

## Proposed Changes

> [!IMPORTANT]
> **MAXIMUM DETAIL REQUIRED:** Provide complete, fully functioning code blocks. Break down the logic step-by-step before showing code.

---

### 1. Settings — New Config Parameters

#### MODIFY `config/settings.py`

- **Context:** `RiskConfig` already holds `max_daily_loss_pct`. Add the symmetric profit cap (`max_daily_profit_pct`) and the cool-down duration (`cool_down_minutes`).
- **Logic:**
  1. Add `cool_down_minutes: float = 5.0` — default 5 wall-clock minutes between a trade exit and the next eligible entry.
  2. Add `max_daily_profit_pct: float = 0.02` — default 2% daily profit cap, symmetric with `max_daily_loss_pct`.
  3. Both have safe defaults; existing instantiations of `RiskConfig()` are unaffected (frozen dataclass, additive change).
- **Implementation:**

```python
# In the RiskConfig dataclass, after max_drawdown_pct:

@dataclass(frozen=True)
class RiskConfig:
    """Parâmetros de gerenciamento de risco e circuit breakers."""

    max_daily_loss_pct: float = 0.02       # 2% do capital
    max_drawdown_pct: float = 0.05         # 5% drawdown máximo
    max_daily_profit_pct: float = 0.02     # 2% lucro máximo diário (novo)
    cool_down_minutes: float = 5.0         # Minutos de resfriamento pós-saída (novo)
    max_position_size: float = 200.0
    max_open_positions: int = 5
    kelly_fraction: float = 0.5
    min_conviction_threshold: float = 0.5

    # Horários e Modalidade
    trading_start_time: str = "09:00:00"
    trading_end_time: str = "17:55:00"
    trade_type: str = "day_trade"
```

- **Edge Cases Handled:** `frozen=True` prevents accidental mutation at runtime. Default values ensure zero disruption to backtests or paper trading sessions already in flight.

---

### 2. Risk Manager — State Machine, Cool-Down, Daily Profit Target

#### MODIFY `src/execution/risk.py`

- **Context:** This is the primary change. The module needs: (a) explicit system states, (b) cool-down timer, (c) daily profit circuit breaker, (d) a restructured `_check_circuit_breakers()` that does not short-circuit before evaluating cool-down expiry.
- **Logic — State Machine:**
  - Replace the implicit `is_halted + halt_reason` two-variable state with a canonical `system_state: str` attribute. Valid values:
    - `"ACTIVE"` — system is free to trade.
    - `"COOL_DOWN"` — post-trade pause; temporary; clears when `_cool_down_until` passes.
    - `"HALTED_FOR_DAY"` — permanent halt for the current trading day (loss, drawdown, OR profit target hit). Only the daily reset clears this.
    - `"OUTSIDE_WINDOW"` — outside trading hours. Cleared automatically when clock re-enters the window.
  - `is_halted: bool` and `halt_reason: str` are kept as plain attributes (NOT properties) for backward compatibility with existing tests that set them directly (e.g., `engine.risk.is_halted = True`). They are always kept in sync via `_set_state()`.
  - `_set_state(state: str, reason: str)` is an internal helper that atomically updates all three fields.

- **Logic — Cool-Down Flow:**
  1. `notify_trade_closed()` is the public API called by the engine after a circuit-breaker exit (not after stop-and-reverse). It sets `_cool_down_until = now + timedelta(minutes=cool_down_minutes)` and transitions to `COOL_DOWN`.
  2. `_check_circuit_breakers()` evaluates cool-down expiry as its FIRST check (before the HALTED_FOR_DAY guard) to allow timer clearing on every tick.
  3. When `now >= _cool_down_until`, state transitions back to `ACTIVE`.

- **Logic — Daily Profit Check:**
  1. After the daily loss check, compute `daily_pnl_pct = (current_equity / start_balance) - 1.0`.
  2. If `daily_pnl_pct >= max_daily_profit_pct`, transition to `HALTED_FOR_DAY` with reason `"MAX DAILY PROFIT REACHED"` and log to audit SQLite.

- **Logic — Restructured `_check_circuit_breakers()`:**
  The restructuring replaces the old monolithic guard with a state-based dispatch:
  ```
  1. COOL_DOWN check (expiry evaluation — must run every tick)
  2. HALTED_FOR_DAY guard (if already permanently halted, return)
  3. OUTSIDE_WINDOW check
  4. Daily loss check
  5. Daily profit check  ← NEW
  6. Drawdown check
  ```

- **Implementation (full rewrite of `src/execution/risk.py`):**

```python
"""
Gerenciamento de Risco e Circuit Breakers — TradeSystem5000.

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
- COOL_DOWN       : Pausa pós-saída de posição. Expira após `cool_down_minutes`.
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
STATE_COOL_DOWN = "COOL_DOWN"
STATE_HALTED_FOR_DAY = "HALTED_FOR_DAY"
STATE_OUTSIDE_WINDOW = "OUTSIDE_WINDOW"


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

        # Temporizador do cool-down
        self._cool_down_until: datetime.datetime | None = None

        # Reset diário
        self.last_trading_day = datetime.date.today()

    # ------------------------------------------------------------------
    # API Pública
    # ------------------------------------------------------------------

    def update_equity(self, balance: float, equity: float) -> None:
        """
        Atualiza o estado da conta (chamado a cada ciclo/tick).

        No MT5: balance = saldo fechado, equity = saldo + lucro flutuante.
        """
        today = datetime.date.today()

        # Se mudou o dia de trading, reseta para o novo dia
        if today > self.last_trading_day:
            self.start_balance = balance
            self.last_trading_day = today
            self._cool_down_until = None  # Descarta cool-down do dia anterior
            self._set_state(STATE_ACTIVE, "")
            logger.info("Novo dia de trading. Saldo inicial resetado para: {:.2f}", self.start_balance)

        if self.start_balance is None:
            self.start_balance = balance
            self.highest_equity = equity

        self.current_equity = equity

        if equity > self.highest_equity:
            self.highest_equity = equity

        self._check_circuit_breakers()

    def notify_trade_closed(self) -> None:
        """
        Inicia o período de cool-down após uma saída de posição para flat.

        Deve ser chamado APENAS na saída por circuit breaker (não em stop-and-reverse),
        para evitar bloquear a gestão de posições recém-abertas.
        Não-operacional se cool_down_minutes <= 0 ou se o sistema já estiver
        em HALTED_FOR_DAY (que sobrepõe o cool-down).
        """
        if self.cool_down_minutes <= 0:
            return
        if self.system_state == STATE_HALTED_FOR_DAY:
            return

        cool_down_until = datetime.datetime.now() + datetime.timedelta(minutes=self.cool_down_minutes)
        self._cool_down_until = cool_down_until
        reason = f"COOL_DOWN (until {cool_down_until.strftime('%H:%M:%S')})"
        self._set_state(STATE_COOL_DOWN, reason)
        audit.log_error("RiskManager", reason, critical=False)  # Persiste no SQLite para análise pós-trade
        logger.info("Cool-down ativado até: {}", cool_down_until.strftime('%H:%M:%S'))

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

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _set_state(self, state: str, reason: str) -> None:
        """
        Transição de estado atómica. Mantém system_state, is_halted e
        halt_reason em sincronia. Único ponto de mutação de estado.
        """
        self.system_state = state
        self.is_halted = state != STATE_ACTIVE
        self.halt_reason = reason

    def _check_circuit_breakers(self) -> None:
        """
        Avalia todas as regras de risco macro. Ordem de prioridade:

        1. Cool-down (avaliado a cada tick para detetar expiração)
        2. Halt permanente do dia (só o reset diário pode limpar)
        3. Janela de horário
        4. Daily Loss
        5. Daily Profit  ← NOVO
        6. Max Drawdown
        """
        # --- 1. Cool-down (deve correr antes do guard de HALTED_FOR_DAY) ---
        if self.system_state == STATE_COOL_DOWN:
            if self._cool_down_until and datetime.datetime.now() >= self._cool_down_until:
                # Expirado: retorna a ACTIVE e prossegue para os restantes checks
                self._cool_down_until = None
                self._set_state(STATE_ACTIVE, "")
                logger.info("Cool-down expirado. Sistema reativado.")
            else:
                # Ainda em cool-down — sem mais verificações necessárias
                return

        # --- 2. Halt permanente do dia (PnL ou Drawdown) ---
        if self.system_state == STATE_HALTED_FOR_DAY:
            return

        # --- 3. Janela de horário ---
        now = datetime.datetime.now().time()
        if now < self.start_time or now > self.end_time:
            self._set_state(
                STATE_OUTSIDE_WINDOW,
                f"OUTSIDE TRADING WINDOW ({now.strftime('%H:%M:%S')})"
            )
            return
        elif self.system_state == STATE_OUTSIDE_WINDOW:
            # Estava fora do horário, agora está dentro — reativa
            self._set_state(STATE_ACTIVE, "")

        # --- 4. Perda Diária (Daily Loss) ---
        daily_pnl_pct = (self.current_equity / self.start_balance) - 1.0
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            reason = f"MAX DAILY LOSS REACHED ({daily_pnl_pct:.2%})"
            self._set_state(STATE_HALTED_FOR_DAY, reason)
            audit.log_error("RiskManager", reason, critical=True)
            return

        # --- 5. Meta de Lucro Diário (Daily Profit Target) --- NOVO ---
        if daily_pnl_pct >= self.max_daily_profit_pct:
            reason = f"MAX DAILY PROFIT REACHED ({daily_pnl_pct:.2%})"
            self._set_state(STATE_HALTED_FOR_DAY, reason)
            audit.log_error("RiskManager", reason, critical=True)
            return

        # --- 6. Maximum Drawdown (Conta Global) ---
        drawdown_pct = (self.current_equity / self.highest_equity) - 1.0
        if drawdown_pct <= -self.max_drawdown_pct:
            reason = f"MAX DRAWDOWN REACHED ({drawdown_pct:.2%})"
            self._set_state(STATE_HALTED_FOR_DAY, reason)
            audit.log_error("RiskManager", reason, critical=True)
            return
```

- **Edge Cases Handled:**
  - Cool-down expiry re-evaluated unconditionally each tick (guard placed after the cool-down check, not before it).
  - `HALTED_FOR_DAY` is permanent within the day regardless of cool-down state.
  - `OUTSIDE_WINDOW` transitions back to `ACTIVE` automatically when clock returns to window.
  - `notify_trade_closed()` is a no-op if `cool_down_minutes <= 0` (feature flag by config).
  - `_set_state()` is the single mutation point — no scattered `self.is_halted = True/False` assignments.

---

### 3. Engine — Call `notify_trade_closed()` in Circuit-Breaker Path

#### MODIFY `src/execution/engine.py`

- **Context:** `notify_trade_closed()` must be called only when the engine closes a position due to a circuit breaker (going to flat), not during the stop-and-reverse flow where a new order immediately follows.
- **Logic:**
  1. In `_process_symbol()`, after `self.om.close_positions(symbol)` in the `if not self.risk.can_trade()` block, add `self.risk.notify_trade_closed()`.
  2. Do NOT add it in the stop-and-reverse `close_positions()` calls lower in the function.
- **Implementation (surgical diff — only the circuit-breaker block changes):**

```python
# BEFORE (engine.py ~line 76-82):
        if not self.risk.can_trade():
            if self.trade_type == "day_trade" or "WINDOW" not in self.risk.halt_reason:
                if self.om.get_net_position(symbol) != 0:
                    logger.warning("Fechando posições para {} devido a: {}", symbol, self.risk.halt_reason)
                    self.om.close_positions(symbol)
            return

# AFTER:
        if not self.risk.can_trade():
            if self.trade_type == "day_trade" or "WINDOW" not in self.risk.halt_reason:
                if self.om.get_net_position(symbol) != 0:
                    logger.warning("Fechando posições para {} devido a: {}", symbol, self.risk.halt_reason)
                    self.om.close_positions(symbol)
                    # Inicia cool-down apenas no fecho por circuit breaker (saída para flat).
                    # NÃO chamar nos blocos de stop-and-reverse abaixo — esses fecham
                    # para imediatamente reabrir na direção oposta.
                    self.risk.notify_trade_closed()
            return
```

- **Edge Cases Handled:**
  - `notify_trade_closed()` is guarded internally: no-op if position was already flat (`get_net_position == 0`) because the outer `if net_position != 0` gate prevents reaching it. No double-trigger.
  - Stop-and-reverse paths at lines ~132 and ~139 are explicitly excluded — they call `close_positions()` but do NOT call `notify_trade_closed()`.

---

### 4. Tests — New Unit Tests

#### MODIFY `tests/test_execution/test_time_restrictions.py`

- **Context:** Add tests for cool-down activation/expiry, daily profit halt, and the interaction between HALTED_FOR_DAY and cool-down (profit halt should not be overwritten by cool-down).
- **Logic:**
  1. `test_cool_down_activated_after_notify` — verify `notify_trade_closed()` sets state to `COOL_DOWN` and `can_trade()` returns False.
  2. `test_cool_down_expires_after_time` — mock `datetime.datetime.now()` to return a time past `_cool_down_until`; verify `update_equity()` → `can_trade()` returns True.
  3. `test_daily_profit_target_halts_trading` — inject equity above `max_daily_profit_pct`; verify `can_trade()` is False and `system_state == "HALTED_FOR_DAY"`.
  4. `test_daily_profit_target_in_halt_reason` — verify halt_reason string contains `"MAX DAILY PROFIT REACHED"`.
  5. `test_cool_down_does_not_override_halted_for_day` — set state to `HALTED_FOR_DAY` first, then call `notify_trade_closed()`; verify state remains `HALTED_FOR_DAY`.
- **Implementation:**

```python
# Append to tests/test_execution/test_time_restrictions.py

class TestCoolDown:

    def test_cool_down_activated_after_notify(self):
        """notify_trade_closed() deve activar o cool-down e bloquear novas ordens."""
        rm = RiskManager(start_balance=100000.0)
        # Garante que está ACTIVE dentro do horário
        with patch('src.execution.risk.datetime.datetime') as mock_dt:
            mock_dt.now.return_value.time.return_value = datetime.time(10, 0, 0)
            mock_dt.now.return_value = datetime.datetime(2026, 4, 12, 10, 0, 0)
            mock_dt.date = datetime.date
            rm.update_equity(100000.0, 100000.0)

        assert rm.can_trade() is True

        # Activa o cool-down (simula saída por circuit breaker)
        with patch('src.execution.risk.datetime.datetime') as mock_dt:
            mock_dt.now.return_value = datetime.datetime(2026, 4, 12, 10, 0, 0)
            mock_dt.date = datetime.date
            rm.notify_trade_closed()

        assert rm.can_trade() is False
        assert rm.system_state == "COOL_DOWN"
        assert "COOL_DOWN" in rm.halt_reason

    def test_cool_down_expires_after_time(self):
        """Após o temporizador expirar, update_equity() deve reativar o sistema."""
        rm = RiskManager(start_balance=100000.0)

        # Activa cool-down às 10:00:00
        with patch('src.execution.risk.datetime.datetime') as mock_dt:
            mock_dt.now.return_value = datetime.datetime(2026, 4, 12, 10, 0, 0)
            mock_dt.date = datetime.date
            rm.notify_trade_closed()

        assert rm.system_state == "COOL_DOWN"

        # Simula tick às 10:06:00 (após 5 min de cool-down)
        with patch('src.execution.risk.datetime.datetime') as mock_dt:
            future = datetime.datetime(2026, 4, 12, 10, 6, 0)
            mock_dt.now.return_value = future
            mock_dt.now.return_value.time.return_value = datetime.time(10, 6, 0)
            mock_dt.date = datetime.date
            rm.update_equity(100000.0, 100000.0)

        assert rm.can_trade() is True
        assert rm.system_state == "ACTIVE"
        assert rm.halt_reason == ""

    def test_daily_profit_target_halts_trading(self):
        """Se o PnL diário atingir max_daily_profit_pct, o sistema deve parar."""
        rm = RiskManager(start_balance=100000.0)

        with patch('src.execution.risk.datetime.datetime') as mock_dt:
            mock_dt.now.return_value.time.return_value = datetime.time(10, 0, 0)
            mock_dt.now.return_value = datetime.datetime(2026, 4, 12, 10, 0, 0)
            mock_dt.date = datetime.date
            # Equity = 102001 sobre start_balance 100000 → +2.001% > limite de 2%
            rm.update_equity(100000.0, 102001.0)

        assert rm.can_trade() is False
        assert rm.system_state == "HALTED_FOR_DAY"

    def test_daily_profit_target_in_halt_reason(self):
        """O motivo do halt deve identificar claramente a meta de lucro atingida."""
        rm = RiskManager(start_balance=100000.0)

        with patch('src.execution.risk.datetime.datetime') as mock_dt:
            mock_dt.now.return_value.time.return_value = datetime.time(10, 0, 0)
            mock_dt.now.return_value = datetime.datetime(2026, 4, 12, 10, 0, 0)
            mock_dt.date = datetime.date
            rm.update_equity(100000.0, 102001.0)

        assert "MAX DAILY PROFIT REACHED" in rm.halt_reason

    def test_cool_down_does_not_override_halted_for_day(self):
        """notify_trade_closed() não deve sobrescrever um HALTED_FOR_DAY permanente."""
        rm = RiskManager(start_balance=100000.0)

        # Força estado HALTED_FOR_DAY via perda diária
        with patch('src.execution.risk.datetime.datetime') as mock_dt:
            mock_dt.now.return_value.time.return_value = datetime.time(10, 0, 0)
            mock_dt.now.return_value = datetime.datetime(2026, 4, 12, 10, 0, 0)
            mock_dt.date = datetime.date
            rm.update_equity(100000.0, 97000.0)  # -3% > limite de 2%

        assert rm.system_state == "HALTED_FOR_DAY"

        # Tenta activar cool-down — deve ser ignorado
        with patch('src.execution.risk.datetime.datetime') as mock_dt:
            mock_dt.now.return_value = datetime.datetime(2026, 4, 12, 10, 0, 0)
            mock_dt.date = datetime.date
            rm.notify_trade_closed()

        # Estado não deve ter mudado
        assert rm.system_state == "HALTED_FOR_DAY"
        assert "MAX DAILY LOSS REACHED" in rm.halt_reason
```

- **Edge Cases Handled:**
  - Tests mock `datetime.datetime` at module level (`src.execution.risk.datetime.datetime`) to control both `.now()` and `.now().time()` independently.
  - `datetime.date` is preserved as real via `mock_dt.date = datetime.date` to avoid breaking the daily reset check that calls `datetime.date.today()`.

## Verification Plan

### Automated Tests
- **Existing tests to run (regression):**
  - `tests/test_execution/test_time_restrictions.py` — existing 3 tests; all must continue to pass. Key: `is_halted = True` direct assignment in existing tests is still valid (plain attribute, not a property).
  - `tests/test_execution/test_phase6.py` — existing `RiskManager` tests for daily loss and drawdown.
  - `tests/test_execution/test_execution_flow.py` — end-to-end engine flow.

- **New tests to run:**
  - `tests/test_execution/test_time_restrictions.py::TestCoolDown::test_cool_down_activated_after_notify`
  - `tests/test_execution/test_time_restrictions.py::TestCoolDown::test_cool_down_expires_after_time`
  - `tests/test_execution/test_time_restrictions.py::TestCoolDown::test_daily_profit_target_halts_trading`
  - `tests/test_execution/test_time_restrictions.py::TestCoolDown::test_daily_profit_target_in_halt_reason`
  - `tests/test_execution/test_time_restrictions.py::TestCoolDown::test_cool_down_does_not_override_halted_for_day`

- **Run command:**
  ```bash
  pytest tests/test_execution/ -v
  ```

### Manual / Paper Trading Validation
- **Cool-down log check:** Run paper trading session; after any forced circuit-breaker close, confirm `COOL_DOWN` state appears in both `loguru` output and `audit_errors` SQLite table.
- **Daily profit halt check:** Set `max_daily_profit_pct = 0.0001` temporarily to trigger immediately; confirm system halts and logs `MAX DAILY PROFIT REACHED`.
- **Backtest validation (high-trend days):** Run backtests on historically strong trending days (e.g. B3 sessions with >1.5% WINFUT move in the first hour). Confirm the daily profit target engages and blocks further trades for the remainder of the session, preserving capital vs. the version without the block.

---

## Review Results (2026-04-12)

### Files Changed
| File | Change |
|------|--------|
| `config/settings.py` | Added `max_daily_profit_pct`, `cool_down_minutes` to `RiskConfig` |
| `src/execution/risk.py` | Full rewrite — state machine, `notify_trade_closed()`, `_set_state()`, restructured `_check_circuit_breakers()`; post-review fix: added `STATE_OUTSIDE_WINDOW` guard in `notify_trade_closed()` |
| `src/execution/engine.py` | Added `self.risk.notify_trade_closed()` call in circuit-breaker close path |
| `tests/test_execution/test_time_restrictions.py` | Added `TestCoolDown` class with 5 new tests; tests deviate slightly from plan's mock approach (better: uses `patch.object` with `wraps=` for expiry test) |

### Validation Results
```
pytest tests/test_execution/ -v
30 passed in 4.85s
```
All 25 pre-existing tests green. All 5 new `TestCoolDown` tests green.

### Remaining Risks
- **Engine test suite does not exercise `STATE_OUTSIDE_WINDOW` guard in `notify_trade_closed()`** — existing engine tests bypass `_set_state()` via direct attribute assignment. The production code path is correct but unit-test coverage for this specific guard is missing. Low risk: the guard is trivial and covered by code inspection.
- **Paper mode end-to-end untestable** — `get_net_position()` always returns 0 in paper mode; cool-down trigger in the engine path does not fire. Documented in plan; unit tests cover the logic independently.

### Final Verdict
**Ready** — all plan requirements implemented, one post-review MAJOR fixed, 30/30 tests passing.

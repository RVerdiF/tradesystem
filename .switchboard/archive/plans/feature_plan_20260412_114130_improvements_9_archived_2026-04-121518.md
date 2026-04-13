# Improvements 9 — Bet Sizing: Abandon Binary Allocation

## Goal
Abandon the system's binary position allocation (100% or 0%) and replace it with a proportional Bet Sizing mechanism. The Meta-Model's continuous probability output must flow through Kelly Criterion sizing into the OrderManager, with any lot below a configurable conviction threshold resulting in a zero position (no trade sent to the broker).

## User Review Required
> [!NOTE]
> - **Breaking change in `AsyncTradingEngine`**: The `meta_label == 1` binary gate is removed. The system will now trade with fractional lots. Paper-trade for at least one full session before enabling live mode.
> - **`max_position` parameter in `AsyncTradingEngine.__init__`**: Previously treated as "lotes máximos" and implicitly 1. Now it is the absolute ceiling passed to `discretize_bet`. Confirm its value matches the intended max leverage before deployment.
> - **Minimum lot enforcement removed**: The `max(1, ...)` guard is deleted from engine.py. The system CAN and WILL send zero-lot signals when conviction is low. Verify `order_manager.send_market_order` guard (Step 4) is deployed before going live.
> - No manual DB migration required.

## Complexity Audit
### Routine
- Add `min_conviction_threshold: float` field to `RiskConfig` in `config/settings.py`
- Add `apply_conviction_threshold()` helper function to `src/modeling/bet_sizing.py`
- Add zero-volume early-return guard to `OrderManager.send_market_order` in `src/execution/order_manager.py`
- Write unit tests for conviction threshold in `tests/test_modeling/test_phase4.py`
- Write unit test for zero-volume guard in `tests/test_execution/test_execution_flow.py`

### Complex / Risky
- **`src/execution/engine.py`** — Removing the binary `meta_label == 1` gate and the `max(1, ...)` floor changes order-flow behavior. A subtle bug: if `model_pipeline` returns `kelly_fraction = 0.0` AND `alpha_side != 0`, the engine must **not** call `close_positions` (it would close an existing profitable position without opening a replacement, causing unnecessary churn and commissions).
- **`src/execution/engine.py`** — The `discretize_bet` function signature accepts a `pd.Series`, not a scalar float. The engine works with scalar `kelly_f` values from the pipeline. A scalar-compatible path must be explicitly used; calling `discretize_bet` on a scalar needs `.iloc[0]` or a direct scalar arithmetic path — **do not silently pass a float where a Series is expected**.
- **`src/modeling/bet_sizing.py`** — The `apply_conviction_threshold` function must zero the *probability* before Kelly is computed (pre-Kelly), NOT clip the kelly fraction *after* (post-Kelly). These produce different outputs; the threshold semantics must be on the raw probability.

## Edge-Case & Dependency Audit
- **Race Conditions:** The engine's `_process_symbol` is async. `close_positions` is called before `send_market_order`. If `kelly_f = 0` and the engine incorrectly calls `close_positions` before checking, a profitable open position will be closed with no replacement. The guard must check `target_volume > 0` BEFORE calling `close_positions`.
- **Security:** No new attack surface introduced. Input from `model_pipeline` is treated as trusted internal output.
- **Side Effects:** `discretize_bet` emits a logger.info for series with `len > 1`. When called per-symbol per-tick in the engine with a scalar (wrapped in a Series of length 1), this log line will be suppressed (correct: the guard is `len > 1`). No log spam issue.
- **Dependencies & Conflicts:** No other plans exist on the Kanban board at this time (single plan in `.switchboard/plans/`). The `OptimizationConfig.meta_threshold_range` in `config/settings.py` (line 218) already uses the range `(0.60, 0.75)` for the threshold — this confirms `min_conviction_threshold` defaults should live near 0.5–0.65. The Optuna tuner must be updated separately to sweep `min_conviction_threshold` as a hyperparameter (out of scope for this plan; flag for future work).

## Adversarial Synthesis

### Grumpy Critique

*[GRUMPY PRINCIPAL ENGINEER ENTERS THE CHAT, SLAMMING HIS KEYBOARD]*

Oh, WONDERFUL. Another plan that starts by announcing it will "replace binary allocation with Kelly sizing" — as if no one has looked at the code. LOOK AT THE CODE. `bet_sizing.py` ALREADY EXISTS. `compute_kelly_fraction` ALREADY EXISTS. `discretize_bet` ALREADY EXISTS. The engine ALREADY reads `kelly_fraction` from the pipeline. We haven't been doing binary allocation in the model layer for months. The plan is solving a problem it hasn't even fully diagnosed.

But fine. Let's talk about what IS actually broken:

**Bug #1 — The `max(1, ...)` abomination.** Line 123 and 134 of `engine.py`: `target_volume = float(max(1, int(round(kelly_f * self.max_position))))`. BEAUTIFUL. So you've spent weeks training a Kelly model that can output 0.02 (Kelly says "this is a terrible trade, bet 2%") and we're just going to round that to... 1 FULL LOT? The Kelly output is completely ignored for the actual sizing decision. The model might as well output a coin flip. This is the real problem and the plan buries it.

**Bug #2 — close_positions BEFORE the volume check.** The current flow in `_process_symbol` calls `self.om.close_positions(symbol)` BEFORE computing `target_volume`. The plan says "add a zero check" but if you add it AFTER the close call, you'll close a perfectly good existing position for a zero-volume signal. Now you're flat with nowhere to go. Brilliant. That's money left on the table.

**Bug #3 — scalar vs Series type confusion.** `discretize_bet` is designed for `pd.Series` batch operations. The engine works with a single scalar `kelly_f`. If some well-intentioned developer calls `discretize_bet(pd.Series([kelly_f]), max_position=N).iloc[0]` that works fine — but if they call `discretize_bet(kelly_f, max_position=N)` directly, they'll get a `AttributeError` on the `.clip()` call because numpy scalars don't have `.index`. The plan says nothing about this.

**Bug #4 — hardcoded 0.5 threshold.** Line 107 of engine.py: `meta_label = 1 if signal_data.get("meta_prob", 0.0) >= 0.5 else 0`. The plan says "make the threshold configurable" but hasn't connected it to the `OptimizationConfig.meta_threshold_range` which already exists and already sweeps `(0.60, 0.75)`. So now you'll have TWO threshold parameters doing THE SAME THING. Pick one system.

**Missing: the `OrderManager` zero-volume backstop.** If someone calls `send_market_order(symbol, "buy", 0.0)` in live mode, MT5 will happily reject it with `retcode=10013` (invalid volume), `wait_order_result` logs it as a critical error, and the system looks like it's on fire even though it was "correct" behavior. Add a clean early-return guard so zero-volume is a silent no-op, not an alarm.

*[SLAMS DOOR]*

### Balanced Response

Grumpy is correct on all five points. The plan document was underspecified and didn't fully audit the existing code before prescribing changes. Here is how the implementation steps address every concern:

1. **`max(1, ...)` removal** is the primary and most impactful change. It is explicitly addressed in Step 3.
2. **`close_positions` guard ordering** is fixed by restructuring the engine logic: compute `target_volume` FIRST, then only call `close_positions` when `target_volume > 0`.
3. **Scalar vs Series** is resolved by using a direct scalar arithmetic path in the engine instead of calling `discretize_bet`. The engine will use `int(round(kelly_f * self.max_position))` after the threshold filter, with explicit `max(0, ...)` instead of `max(1, ...)`.
4. **Threshold duplication** is resolved by making `meta_label` computation use `risk_config.min_conviction_threshold` (the new config field), making it consistent with optimization sweeps. The `OptimizationConfig.meta_threshold_range` documents the search space, and `RiskConfig.min_conviction_threshold` holds the production default.
5. **`OrderManager` zero-volume backstop** is added as an explicit defensive guard (Step 4).

---

## Proposed Changes

> [!IMPORTANT]
> **MAXIMUM DETAIL REQUIRED:** Provide complete, fully functioning code blocks. The changes below are presented as exact search/replace blocks. Every step is self-contained.

---

### Step Group A — Low Complexity (Routine)

---

### A1. Configuration — Add `min_conviction_threshold`
#### MODIFY `config/settings.py`

- **Context:** The engine currently hardcodes `>= 0.5` as the binary threshold to convert `meta_prob` into `meta_label`. Making this configurable allows the Optuna tuner to sweep it and lets operators tune production without code changes.
- **Logic:**
  1. Add `min_conviction_threshold: float = 0.5` as a new field in the `RiskConfig` dataclass.
  2. The default of `0.5` preserves the existing (hardcoded) behavior — no regression.
  3. This field will be read by `engine.py` in Step B1.

- **Implementation:**

```python
# SEARCH (in class RiskConfig, after kelly_fraction line):
    kelly_fraction: float = 0.5            # Kelly fracionário (50%)
    
    # Horários e Modalidade

# REPLACE WITH:
    kelly_fraction: float = 0.5            # Kelly fracionário (50%)
    min_conviction_threshold: float = 0.5  # Limiar mínimo de probabilidade do Meta-Model para operar
    
    # Horários e Modalidade
```

- **Edge Cases Handled:** Default value of 0.5 is backward-compatible. The field is `frozen=True` (immutable dataclass), so it cannot be accidentally mutated at runtime.

---

### A2. Bet Sizing Module — Add `apply_conviction_threshold` helper
#### MODIFY `src/modeling/bet_sizing.py`

- **Context:** The conviction threshold logic must live in the bet sizing module (not in the engine) to keep concerns separated and to allow backtest pipelines to apply the same filter independently. The threshold must operate on the raw probability BEFORE Kelly is computed, not on the Kelly output — this is the semantically correct placement (a probability below threshold means "no edge", not "Kelly says bet small").
- **Logic:**
  1. Add a new `apply_conviction_threshold(prob_win, threshold)` function.
  2. The function accepts either a scalar float or a pd.Series/np.ndarray.
  3. For any element where `prob_win < threshold`, the output probability is set to 0.0.
  4. A probability of 0.0 fed into `compute_kelly_fraction` will produce a Kelly fraction of `0.0 - 1.0/odds = -1.0`, which is clipped to 0.0. So the chain `apply_conviction_threshold → compute_kelly_fraction` correctly produces zero sizing.

- **Implementation — add after the `discretize_bet` function:**

```python
def apply_conviction_threshold(
    prob_win: pd.Series | np.ndarray | float,
    threshold: float | None = None,
) -> pd.Series | np.ndarray | float:
    """
    Zera a probabilidade de entradas abaixo do limiar de convicção.

    Opera ANTES do cálculo do Kelly. Uma probabilidade zerada produz um
    Kelly negativo que é clipado para 0, resultando em posição zerada.

    Isso separa dois conceitos distintos:
    - Abaixo do threshold: "Não há edge suficiente" → prob = 0 → Kelly = 0 → lote = 0.
    - Acima do threshold: "Edge existe, mas pode ser fraco" → Kelly fraccionário.

    Parameters
    ----------
    prob_win : float, np.ndarray ou pd.Series
        Probabilidade de sucesso prevista pelo Meta-Model.
    threshold : float, optional
        Limiar mínimo de probabilidade. Se não fornecido, busca em risk_config.

    Returns
    -------
    prob_filtered : mesma estrutura de prob_win, com zeros onde abaixo do threshold.
    """
    if threshold is None:
        threshold = risk_config.min_conviction_threshold

    if isinstance(prob_win, pd.Series):
        filtered = prob_win.copy()
        filtered[filtered < threshold] = 0.0
        return filtered
    elif isinstance(prob_win, np.ndarray):
        filtered = prob_win.copy()
        filtered[filtered < threshold] = 0.0
        return filtered
    else:
        # scalar float
        return float(prob_win) if float(prob_win) >= threshold else 0.0
```

Also update the module docstring `Funcionalidades:` block to include the new function:

```python
# SEARCH:
# - **discretize_bet**: Conversão de frações contínuas em lotes operacionais (discretos).

# REPLACE WITH:
# - **discretize_bet**: Conversão de frações contínuas em lotes operacionais (discretos).
# - **apply_conviction_threshold**: Zera probabilidades abaixo do limiar de convicção (pré-Kelly).
```

- **Edge Cases Handled:** Scalar, ndarray, and Series inputs all handled explicitly. Does not mutate in place (uses `.copy()`). Threshold defaults to config so unit tests can override it via `fraction` param.

---

### A3. OrderManager — Zero-Volume Backstop
#### MODIFY `src/execution/order_manager.py`

- **Context:** If `target_volume = 0` reaches `send_market_order` (e.g., due to a bug in the engine), MT5 in live mode returns `retcode=10013` (INVALID_VOLUME), which triggers `audit.log_error(..., critical=True)`. This creates false alarms in monitoring. A zero-volume order is legitimate "no-op" behavior in the new sizing scheme and should be handled cleanly.
- **Logic:**
  1. Add a guard at the top of `send_market_order` before any MT5 calls.
  2. If `volume <= 0`, log at DEBUG level (not WARNING — it is expected behavior) and return `True` (no error, order just wasn't needed).

- **Implementation:**

```python
# SEARCH (in send_market_order, after the docstring, before the mode check):
        # Proteção: só no modo live tenta enviar ordens reais
        if execution_config.mode != "live":

# REPLACE WITH:
        # Proteção: lote zero é comportamento legítimo do Bet Sizing (sem convicção suficiente).
        # Não constitui erro — retorna True silenciosamente para não poluir logs/alertas.
        if volume <= 0:
            logger.debug(
                "[BET SIZING] Lote zero calculado para {} {}. Nenhuma ordem enviada.",
                action.upper(), symbol
            )
            return True

        # Proteção: só no modo live tenta enviar ordens reais
        if execution_config.mode != "live":
```

- **Edge Cases Handled:** Covers both `volume == 0` (expected) and `volume < 0` (defensive against calculation bugs upstream). Paper mode path is untouched. Live mode MT5 call is never reached with zero volume.

---

### Step Group B — Complex / Risky

---

### B1. Execution Engine — Replace Binary Gate with Proportional Sizing
#### MODIFY `src/execution/engine.py`

- **Context:** This is the central change. Two bugs must be fixed simultaneously:
  1. The `meta_label = 1 if ... >= 0.5 else 0` binary gate discards the continuous probability — it is replaced with `min_conviction_threshold` from config.
  2. The `max(1, ...)` floor forces a minimum of 1 lot, defeating the entire purpose of Kelly sizing.
  3. `close_positions` must only be called AFTER confirming `target_volume > 0`, to avoid closing positions for zero-conviction signals.

- **Logic — step by step:**
  1. Import `risk_config` in engine.py (it already imports `execution_config` — just add `risk_config`).
  2. Remove the `meta_label` variable entirely. Instead, derive `target_volume` directly from `kelly_f` and a threshold check.
  3. The threshold check uses `risk_config.min_conviction_threshold`, matching the new config field.
  4. `target_volume = int(round(kelly_f * self.max_position))` — pure proportional sizing, no `max(1, ...)`.
  5. Move `close_positions` call INSIDE the `target_volume > 0` check, so a zero-conviction signal leaves the existing position untouched.

- **Implementation:**

```python
# SEARCH (import line, top of engine.py):
from config.settings import execution_config

# REPLACE WITH:
from config.settings import execution_config, risk_config
```

```python
# SEARCH (in _process_symbol, signal extraction block):
            # Formato esperado: {"side": 1/-1/0, "meta_prob": float, "kelly_fraction": float, "price": float}
            alpha_side = signal_data.get("side", 0)
            meta_label = 1 if signal_data.get("meta_prob", 0.0) >= 0.5 else 0
            kelly_f = signal_data.get("kelly_fraction", 0.0)
            price = signal_data.get("price", 0.0)

            # Rastreia
            if alpha_side != 0:
                audit.log_signal(symbol, alpha_side, meta_label, kelly_f, price)

            # 4. Avalia Ação Direcional
            net_position = self.om.get_net_position(symbol)

            # Exemplo simples de Stop-and-Reverse contínuo:
            if alpha_side == 1 and net_position <= 0 and meta_label == 1 and kelly_f > 0:
                # Comprar! Fecha posicao vendida antes
                self.om.close_positions(symbol)

                # Sizing final (arredondado para o inteiro mais prximo e garantindo lote mnimo 1)
                target_volume = float(max(1, int(round(kelly_f * self.max_position))))

                # Valida usando a posição atual (caso close_positions tenha falhado parcialmente)
                if self.risk.validate_order(abs(self.om.get_net_position(symbol)), target_volume, self.max_position):
                    self.om.send_market_order(symbol, "buy", target_volume)

            elif alpha_side == -1 and net_position >= 0 and meta_label == 1 and kelly_f > 0:
                # Vender!
                self.om.close_positions(symbol)

                target_volume = float(max(1, int(round(kelly_f * self.max_position))))

                if self.risk.validate_order(abs(self.om.get_net_position(symbol)), target_volume, self.max_position):
                    self.om.send_market_order(symbol, "sell", target_volume)

# REPLACE WITH:
            # Formato esperado: {"side": 1/-1/0, "meta_prob": float, "kelly_fraction": float, "price": float}
            alpha_side = signal_data.get("side", 0)
            meta_prob = signal_data.get("meta_prob", 0.0)
            kelly_f = signal_data.get("kelly_fraction", 0.0)
            price = signal_data.get("price", 0.0)

            # Aplica limiar de convicção: probabilidade abaixo do threshold → kelly zerado
            # Isso substitui a conversão binária (meta_label == 1) por uma decisão contínua.
            # meta_prob abaixo do threshold indica ausência de edge suficiente; kelly_f já
            # deveria refletir isso se pipeline estiver correto, mas verificamos meta_prob
            # diretamente aqui como segunda linha de defesa.
            if meta_prob < risk_config.min_conviction_threshold:
                kelly_f = 0.0

            # Sizing proporcional: sem floor de 1 lote — zero é um resultado válido e esperado.
            target_volume = int(round(kelly_f * self.max_position))

            # Rastreia o sinal (incluindo lotes zero para análise de cobertura)
            if alpha_side != 0:
                audit.log_signal(symbol, alpha_side, int(meta_prob >= risk_config.min_conviction_threshold), kelly_f, price)

            # 4. Avalia Ação Direcional
            net_position = self.om.get_net_position(symbol)

            # Stop-and-Reverse proporcional:
            # IMPORTANTE: close_positions é chamado SOMENTE quando target_volume > 0.
            # Um sinal de baixa convicção (lote zero) NÃO fecha posições existentes —
            # isso evita churn e custos desnecessários em períodos de ruído.
            if alpha_side == 1 and net_position <= 0 and target_volume > 0:
                # Comprar! Fecha posição vendida antes, então abre long proporcional.
                self.om.close_positions(symbol)

                if self.risk.validate_order(abs(self.om.get_net_position(symbol)), float(target_volume), self.max_position):
                    self.om.send_market_order(symbol, "buy", float(target_volume))

            elif alpha_side == -1 and net_position >= 0 and target_volume > 0:
                # Vender! Fecha posição comprada antes, então abre short proporcional.
                self.om.close_positions(symbol)

                if self.risk.validate_order(abs(self.om.get_net_position(symbol)), float(target_volume), self.max_position):
                    self.om.send_market_order(symbol, "sell", float(target_volume))
```

- **Edge Cases Handled:**
  - `meta_prob < threshold AND kelly_f > 0`: Caught by threshold check → kelly_f zeroed → target_volume = 0 → no trade, no close.
  - `meta_prob >= threshold AND kelly_f = 0.0`: Kelly already says no edge. target_volume = 0 → no trade (correct).
  - `alpha_side = 0`: Neither branch fires (unchanged behavior).
  - `close_positions` partial failure: The post-close `get_net_position` call inside `validate_order` catches residual exposure (unchanged behavior).

---

### Step Group C — Tests

---

### C1. Bet Sizing Tests — Conviction Threshold
#### MODIFY `tests/test_modeling/test_phase4.py`

- **Context:** The new `apply_conviction_threshold` function needs unit tests covering scalar/Series inputs and both pass/block cases.
- **Logic:** Add a new `TestConvictionThreshold` class to the existing test file.

- **Implementation — append to the end of the file:**

```python
# ---------------------------------------------------------------------------
# Testes — Conviction Threshold
# ---------------------------------------------------------------------------
class TestConvictionThreshold:
    """Testa o filtro de probabilidade pre-Kelly."""

    def test_scalar_above_threshold_passes(self):
        """Probabilidade acima do threshold não é modificada."""
        from src.modeling.bet_sizing import apply_conviction_threshold
        result = apply_conviction_threshold(0.65, threshold=0.5)
        assert result == 0.65

    def test_scalar_below_threshold_zeroed(self):
        """Probabilidade abaixo do threshold é zerada."""
        from src.modeling.bet_sizing import apply_conviction_threshold
        result = apply_conviction_threshold(0.45, threshold=0.5)
        assert result == 0.0

    def test_scalar_exactly_at_threshold_passes(self):
        """Probabilidade exatamente igual ao threshold NÃO é zerada (>= semantics)."""
        from src.modeling.bet_sizing import apply_conviction_threshold
        result = apply_conviction_threshold(0.5, threshold=0.5)
        assert result == 0.5

    def test_series_mixed_values(self):
        """Series com valores acima e abaixo do threshold — apenas os baixos são zerados."""
        from src.modeling.bet_sizing import apply_conviction_threshold
        probs = pd.Series([0.3, 0.55, 0.70, 0.49, 0.60])
        filtered = apply_conviction_threshold(probs, threshold=0.5)
        assert filtered.iloc[0] == 0.0   # 0.3 < 0.5 → zerado
        assert filtered.iloc[1] == 0.55  # passa
        assert filtered.iloc[2] == 0.70  # passa
        assert filtered.iloc[3] == 0.0   # 0.49 < 0.5 → zerado
        assert filtered.iloc[4] == 0.60  # passa

    def test_threshold_to_kelly_chain_produces_zero(self):
        """Probabilidade abaixo do threshold → kelly final deve ser zero."""
        from src.modeling.bet_sizing import apply_conviction_threshold, compute_kelly_fraction
        prob_filtered = apply_conviction_threshold(0.40, threshold=0.5)
        kelly = compute_kelly_fraction(prob_win=prob_filtered, odds=1.0, fraction=1.0)
        # prob=0 → f* = 0 - 1/1 = -1 → clipped to 0
        assert kelly == 0.0

    def test_does_not_mutate_input_series(self):
        """A função não deve alterar a Series original (imutabilidade)."""
        from src.modeling.bet_sizing import apply_conviction_threshold
        original = pd.Series([0.3, 0.7])
        original_copy = original.copy()
        apply_conviction_threshold(original, threshold=0.5)
        pd.testing.assert_series_equal(original, original_copy)
```

---

### C2. Engine Integration — Zero-Lot Behavior Test
#### MODIFY `tests/test_execution/test_execution_flow.py`

- **Context:** Verify that the engine does NOT call `close_positions` or `send_market_order` when `kelly_fraction = 0.0` (low conviction signal). This is the regression test for the `close_positions` ordering bug described in the Grumpy critique.
- **Logic:** Add a new test class using mock pipeline and mock OrderManager.

- **Implementation — append to the end of the file:**

```python
# ---------------------------------------------------------------------------
# Testes — Bet Sizing Integration no Engine
# ---------------------------------------------------------------------------
class TestEngineBetSizing:
    """
    Testa que o engine NÃO fecha posições nem envia ordens quando kelly_fraction = 0
    (sinal de baixa convicção / lote zero).
    """

    def _make_engine(self, pipeline_output: dict):
        """Helper: cria AsyncTradingEngine com pipeline mockada."""
        from src.execution.engine import AsyncTradingEngine
        pipeline = MagicMock(return_value=pipeline_output)
        engine = AsyncTradingEngine(
            model_pipeline=pipeline,
            symbols=["WIN$N"],
            max_position=5,
        )
        return engine

    @pytest.mark.asyncio
    async def test_zero_kelly_does_not_close_or_send(self):
        """
        Quando kelly_fraction=0 e meta_prob=0.3 (abaixo do threshold),
        close_positions e send_market_order NÃO devem ser chamados.
        """
        from src.execution.engine import AsyncTradingEngine

        signal = {
            "side": 1,          # Alpha quer comprar
            "meta_prob": 0.30,  # Abaixo do threshold padrão de 0.50
            "kelly_fraction": 0.0,
            "price": 130000.0,
        }

        engine = self._make_engine(signal)

        with patch.object(engine.om, "get_net_position", return_value=-1.0):
            with patch.object(engine.om, "close_positions") as mock_close:
                with patch.object(engine.om, "send_market_order") as mock_send:
                    with patch.object(engine.risk, "can_trade", return_value=True):
                        # Simula dados válidos para não cair no guard de empty/len<50
                        fake_df = pd.DataFrame(
                            {"open": [1]*60, "high": [1]*60, "low": [1]*60,
                             "close": [1]*60, "tick_volume": [1]*60},
                            index=pd.date_range("2024-01-01", periods=60, freq="1min")
                        )
                        engine.model_pipeline = MagicMock(return_value=signal)
                        with patch("src.execution.engine.mt5") as mock_mt5:
                            mock_mt5.copy_rates_from_pos.return_value = fake_df.reset_index().to_dict("records")
                            with patch("src.execution.engine.execution_config") as mock_cfg:
                                mock_cfg.mode = "paper"  # Não tenta conexão MT5 real
                                await engine._process_symbol("WIN$N")

                        mock_close.assert_not_called()
                        mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonzero_kelly_sends_proportional_volume(self):
        """
        Quando kelly_fraction=0.4 e max_position=5, deve enviar volume=2 lotes
        (round(0.4 * 5) = 2), NÃO forçar mínimo de 1 nem máximo estático.
        """
        signal = {
            "side": 1,
            "meta_prob": 0.65,  # Acima do threshold
            "kelly_fraction": 0.4,
            "price": 130000.0,
        }

        from src.execution.engine import AsyncTradingEngine
        engine = self._make_engine(signal)

        with patch.object(engine.om, "get_net_position", return_value=0.0):
            with patch.object(engine.om, "close_positions"):
                with patch.object(engine.om, "send_market_order") as mock_send:
                    with patch.object(engine.risk, "can_trade", return_value=True):
                        with patch.object(engine.risk, "validate_order", return_value=True):
                            with patch("src.execution.engine.execution_config") as mock_cfg:
                                mock_cfg.mode = "paper"
                                engine.model_pipeline = MagicMock(return_value=signal)
                                fake_df = pd.DataFrame(
                                    {"open": [1]*60, "high": [1]*60, "low": [1]*60,
                                     "close": [1]*60, "tick_volume": [1]*60},
                                    index=pd.date_range("2024-01-01", periods=60, freq="1min")
                                )
                                with patch("src.execution.engine.mt5") as mock_mt5:
                                    mock_mt5.copy_rates_from_pos.return_value = fake_df.reset_index().to_dict("records")
                                    await engine._process_symbol("WIN$N")

                        # 0.4 * 5 = 2.0 → round → 2 lotes
                        call_args = mock_send.call_args
                        if call_args:
                            assert call_args[0][2] == 2.0  # volume argument
```

---

## Verification Plan

### Automated Tests
1. **Run existing bet sizing suite** — `pytest tests/test_modeling/test_phase4.py::TestBetSizing -v`
   - All 4 existing tests must continue to pass (no regression in `compute_kelly_fraction` / `discretize_bet`).
2. **Run new conviction threshold suite** — `pytest tests/test_modeling/test_phase4.py::TestConvictionThreshold -v`
   - All 6 new tests must pass.
3. **Run engine integration suite** — `pytest tests/test_execution/test_execution_flow.py::TestEngineBetSizing -v`
   - Verify zero-Kelly no-op behavior and proportional volume dispatch.
4. **Full test suite** — `pytest --tb=short` — no regressions across all phases.

### Validation de Backtest (Out-of-Sample)
- Run a backtest comparing the binary-gated system vs. the proportional system on an out-of-sample period.
- Specifically check: periods where `meta_prob` was between 0.50 and 0.65 — the new system should size down (fractional lot) rather than deploying full 1 lot. Confirm drawdown reduction during noisy periods.

### Auditoria de Logs em Paper Trading
- Deploy in `mode = "paper"` for one full trading session.
- Check logs for:
  - `[BET SIZING] Lote zero calculado` — confirms zero-lot guard firing correctly.
  - `Discretized Pos:` — confirms sizing is proportional, not uniform.
  - Absence of `Ordem rejeitada. Retcode:` — confirms no invalid-volume errors reaching MT5.

---

## Open Questions — Resolved

| Questão | Resolução |
|---|---|
| Qual fórmula de Bet Sizing usar? | Half-Kelly já implementado em `bet_sizing.py` via `risk_config.kelly_fraction = 0.5`. Sem mudança. |
| Qual o threshold de corte ideal? | `RiskConfig.min_conviction_threshold = 0.5` como default de produção. O Optuna já tem `meta_threshold_range = (0.60, 0.75)` para otimização — ambos coexistem. O tuner deve ser atualizado para escrita do threshold otimizado no config (fora de escopo deste plano; registrar como follow-up). |
| Como lidar com lotes fracionários da B3? | `discretize_bet` com `step_size=1` já arredonda para inteiro. O engine usa `int(round(...))` diretamente. Minicontratos WIN (step=1) e contratos cheios WDO (step=1) são ambos cobertos. |

---

## Review & Validation Results (2026-04-12)

### Files Changed
| File | Change |
|---|---|
| `src/execution/engine.py` | **BUG FIX**: Added `risk_config` to import line (was missing — would NameError in production) |
| `tests/test_execution/test_execution_flow.py` | **BUG FIX**: Both `TestEngineBetSizing` tests changed from paper-mode (early-return, hollow) to live-mode with proper unix-timestamp fake records; removed `if call_args:` optional guard replaced with hard assertions |

### Issues Found & Fixed
| Severity | Finding | Resolution |
|---|---|---|
| CRITICAL | `engine.py` used `risk_config` at lines 115/123 but only imported `execution_config` — NameError in any live signal-processing path | Fixed: `from config.settings import execution_config, risk_config` |
| MAJOR | Both engine integration tests patched `mode="paper"` → `df_snapshot = pd.DataFrame()` (empty) → early return on line 99 → tests never reached conviction threshold or `target_volume` logic — passed vacuously | Fixed: mode set to `"live"`, fake records use unix `"time"` field, `audit` patched to avoid SQLite side effects |
| MAJOR | `test_nonzero_kelly_sends_proportional_volume` used `if call_args:` — optional assertion that silently passed when `send_market_order` was never called | Fixed: `mock_send.assert_called_once()` + `assert call_args is not None` + hard volume check |

### Validation Results
```
tests/test_modeling/test_phase4.py::TestBetSizing — 4 passed ✓
tests/test_modeling/test_phase4.py::TestConvictionThreshold — 6 passed ✓
tests/test_execution/test_execution_flow.py::TestEngineBetSizing — 2 passed ✓
Full suite: 213 passed, 0 failed, 0 regressions ✓
```

### Remaining Risks
- **Optuna tuner not updated**: `meta_threshold_range` in `OptimizationConfig` documents the search space but the tuner does not yet write the optimized threshold back into `RiskConfig.min_conviction_threshold`. Flagged as follow-up.
- **Paper mode data path**: `_process_symbol` in `mode="paper"` always produces an empty DataFrame and returns early — no signal is ever processed in paper mode. This is a pre-existing limitation (not introduced by this plan) but should be addressed before paper trading validation can be meaningful.

### Verdict: ✅ READY

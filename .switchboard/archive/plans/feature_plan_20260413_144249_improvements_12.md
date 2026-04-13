# Improvements 12.

## Goal
Fix the structural lookahead/underestimation bug in `src/labeling/triple_barrier.py` by having Take Profit and Stop Loss barriers evaluated against **intrabar High/Low** (not just the close), and by returning the **barrier return** (not the close-based return) when a barrier is crossed — so the backtest's label distribution and `ret` column faithfully reflect realistic fills. Vertical barrier (time stop) continues to use close.

## Metadata
**Tags:** backend, bugfix
**Complexity:** High

## User Review Required
> [!NOTE]
> - **Silent behaviour change:** existing label distributions (+1/-1/0) and `ret` magnitudes will shift. Any cached meta-labeler models, Optuna studies, or saved artefacts trained on the current buggy labels must be **retrained / invalidated** before comparing Sharpe deltas.
> - **API change (breaking internal callers):** `apply_triple_barrier()` and `get_labels()` acquire two new required keyword arguments: `high_prices: pd.Series` and `low_prices: pd.Series`. Every call site in `src/main_backtest.py`, `src/main_execution.py`, and all tests must be updated in the same commit.
> - **Numba kernel signature change:** `_find_first_touch` and `_find_dynamic_touch` gain `high_values: np.ndarray` and `low_values: np.ndarray` parameters. The `@njit` cache will recompile on first run.
> - **Clarification (return sign convention):** returns are stored in *side-adjusted* form (`ret * side`). For Long (`side=+1`) the stored `ret` equals the raw price return; for Short (`side=-1`) the stored `ret` is the negated raw price return. All barrier comparisons below operate on side-adjusted returns, consistent with the existing code.
> - **Clarification (barrier return vs close return):** on a PT/SL cross the function now returns exactly `upper` / `lower` (in side-adjusted terms). On vertical expiry, the close-based return at `end` is preserved, matching the existing `final_ret` semantics.
> - **Clarification (same-bar ambiguity):** if a single bar's range `[low, high]` crosses **both** `upper` and `lower` the kernel cannot know the intrabar ordering. This plan resolves ambiguity conservatively by returning **SL first** (pessimistic fill), matching López de Prado's standard triple-barrier convention. Documented in the docstring.
> - **Clarification (breakeven):** the Breakeven activation check in `_find_dynamic_touch` now triggers when the **intrabar high** (side-adjusted) crosses `upper * be_trigger`, not the close. This matches the corrected exit semantics.

## Background (from original problem statement — preserved verbatim)

Há um erro estrutural crítico na forma como as **saídas (exits)** estão sendo calculadas no seu módulo de Tripla Barreira (`triple_barrier.py`). Essa falha contamina a simulação e é a causa mais provável para que a otimização esteja retornando *Sharpe Ratios* sistematicamente negativos.

### O Problema: Avaliação Limitada ao Preço de Fechamento (`close_values`)
Analisando as funções do kernel otimizado que detectam o toque nas barreiras (`_find_first_touch` e `_find_dynamic_touch`), nota-se que **o sistema está utilizando apenas o preço de fechamento (`close_values`) para verificar as saídas**.

Isso cria três distorções severas que destroem o *Sharpe* da estratégia no backtest:

1. ***Take Profits* (Ganhos) Ignorados:** O código verifica se `ret >= upper` usando o fechamento da barra. Se o mercado atingir o seu *Take Profit* no meio da sessão (máxima da barra), mas recuar e fechar abaixo do alvo, o sistema não registra a vitória.
2. ***Stop Losses* com *Slippage* Catastrófico:** Quando uma barra fecha abaixo do seu *Stop Loss*, o backtest retorna o retorno exato do fechamento daquela barra (`ret = (close_values[i] / entry_price - 1.0) * side`) e não o limite cravado do Stop. Ou seja, se o seu limite era -1% mas a barra derreteu e fechou em -4%, o backtest engole a perda de -4%.
3. ***Stops* Ignorados por Recuperação Intrabarra:** Se o preço violar seu *Stop Loss* durante a barra, mas o fechamento for acima dele, a condição `ret <= lower` será falsa. O trade continuará aberto indevidamente na simulação, acumulando distorções para o futuro.

O mesmo erro lógico se aplica ao **Breakeven**: a ativação da proteção em `_find_dynamic_touch` só ocorre se o fechamento da barra ultrapassar o `be_trigger`. Ele não protege posições que atingiram o alvo e devolveram tudo na mesma barra.

### A Entrada (Entry) Está Correta?
**Sim.** Ao contrário das saídas, a sua mecânica de entrada dos trades está robusta e previne viés de antecipação (*lookahead bias*). O código força explicitamente a passagem dos preços de abertura (`open_prices`) e executa a entrada na abertura da barra seguinte ao sinal da estratégia (`entry_price = open_prices.values[start_loc + 1]`). O problema está restrito unicamente à validação intrabarra dos *exits*.

### Como o Motor de Otimização Reage a Isso
No seu script de otimização (`tuner.py`), existem salvaguardas rigorosas contra *overfitting* e penalizações severas. Se o Meta-Modelo não consegue agregar valor (*Sharpe Lift* <= 0), o *fitness* é dizimado (`fitness *= 0.1`). Como a estratégia base (Alpha) está acumulando perdas irreais pela falha da Tripla Barreira, o Meta-Modelo falha em achar um ganho consistente, levando o Optuna a devolver Sharpes finais negativos ou até mesmo abortar os *trials*.

### Como Corrigir
Para corrigir o comportamento das saídas e alinhar o backtest com a realidade do mercado:

1. Modifique a assinatura de `apply_triple_barrier` e dos métodos `@njit` para receberem também os arrays de **máximas (`high_values`)** e **mínimas (`low_values`)**.
2. Dentro do loop `for i in range(start + 1, end + 1)`, substitua o cálculo único do fechamento. Para posições *Long* (`side == 1`), verifique se o retorno gerado pela Máxima atinge o `upper` (Take Profit) e se a Mínima atinge o `lower` (Stop Loss). Faça o inverso para as posições *Short* (`side == -1`).
3. Em vez de retornar `ret` calculado sobre o preço da barra, **retorne o valor estrito da barreira (`upper` ou `lower`)** sempre que houver um cruzamento. O retorno do fechamento (`final_ret`) só deve ser utilizado quando ocorre o esgotamento do tempo (Barreira Vertical).

## Complexity Audit

### Routine
- Add `high` / `low` parameters to `create_events` is **not required** (it only produces timestamps, targets and sides — barrier evaluation happens later).
- Thread `high_prices` and `low_prices` through `apply_triple_barrier` and `get_labels` signatures.
- Update call sites in `src/main_backtest.py:366` and `src/main_execution.py:173` to pass `df["high"]` / `df["low"]` (or `df_aligned["high"]`/`df_aligned["low"]`).
- Update tests in `tests/test_labeling/test_triple_barrier.py`, `tests/test_labeling/test_phase3.py`, `tests/test_labeling/test_dynamic_barrier.py` to pass `high_prices=close, low_prices=close` (synthetic OHLC) where real OHLC is not constructed — preserves backwards-compatible behaviour for existing assertions.
- Add new unit tests that deliberately construct bars with wide intrabar ranges to exercise the new kernel behaviour (PT-only hit, SL-only hit, ambiguous same-bar PT+SL, BE activation on high-but-close-reverts).

### Complex / Risky
- **Numba `@njit` kernel rewrite** (`_find_first_touch`, `_find_dynamic_touch`): dual-side comparison logic + barrier-return substitution + ambiguous-bar ordering (SL-first). The kernel is the hottest loop in labeling; a subtle off-by-one or sign flip will silently bias labels.
- **Silent behaviour change**: label distribution and `ret` column shift for every historical event. No test asserts absolute counts today (`test_get_labels` only checks `.empty` and column presence), so regressions land green — manual validation against known-answer fixtures is required.
- **Meta-labeler invalidation**: cached `.joblib` / `.pkl` models trained against the buggy labels become untrustworthy; Optuna studies (`tuner.py`) must be re-run after the fix.
- **Breakeven interaction**: the BE trigger now fires on `high` (long) or `low` (short), but the BE SL value (`0.0001` side-adjusted) is unchanged. Must verify that BE→SL hit within the *same bar* that activated BE is handled deterministically (BE activation first, then SL check).

## Edge-Case & Dependency Audit
- **Race Conditions:** None. The kernel is per-event serial; `@njit` routines hold no shared state.
- **Security:** None. No auth/crypto/permissions involved.
- **Side Effects:**
  - `@njit` recompilation on first call after deploy (cold-start latency spike for one run; then cached).
  - Numba cache on disk (`__pycache__/*.nbi`) contains the old signature — delete or rely on Numba's hash-based invalidation.
  - Call sites that do not yet pass `high`/`low` will raise `TypeError` at import of new signature if required kwargs are enforced. Plan enforces required kwargs (no silent `None` fallback) to prevent accidental close-only regressions.
  - If `high < close` or `low > close` on malformed data, the current logic is still deterministic (barrier checks still fire), but upstream data quality is presumed sane; no new guard is added here.
  - NaN in `high`/`low`: if either is NaN for a given bar, `ret_hi`/`ret_lo` become NaN and the `>=`/`<=` comparisons evaluate to False — bar is silently skipped. This matches existing close-NaN behaviour. Document in docstring; do not add explicit NaN handling (same policy as current close loop).
- **Dependencies & Conflicts:**
  - Only one other plan in the Kanban folder: `feature_plan_20260412_120509_improvements_10.md` (Risk Manager cool-down / daily-profit). That plan touches `config/settings.py`, `src/execution/risk.py`, `src/execution/engine.py`, `tests/test_execution/`. **Zero file overlap** with this plan's labeling/backtest scope. Safe to land concurrently.
  - Downstream consumers of `get_labels` output (`src/labeling/meta_labeling.py`, `src/main_backtest.py`, `src/main_execution.py`, `src/optimization/tuner.py`) consume the `ret`/`label`/`barrier_type` columns as-is — no schema change, only value change.

## Adversarial Synthesis

### Grumpy Critique
*Oh, marvellous. We're finally going to let the triple barrier see the candle's **range** instead of pretending the world collapses to a single close print. Only took a few months of negative Sharpes. Let me count the landmines in this "simple fix":*

1. **"Just return `upper` on a PT cross" — you *do* realise the stored `ret` is side-adjusted, right?** If some poor soul later pulls that column and multiplies by `side` thinking it's a raw price return, they'll double-sign themselves into oblivion. You'd better pin this convention in the docstring *and* leave the comment in the kernel, or the next person to touch this will regress it inside a week.
2. **Same-bar double-hit.** A 5-minute bar with a 2% range can absolutely cross both PT and SL. You're asserting "SL first" — fine, that's the LdP convention — but unless you make that choice **explicit in the kernel with a comment and a unit test**, someone will "optimise" it to PT-first in six months and silently make every backtest look like a hedge-fund pitch deck.
3. **Numba `@njit` cache invalidation.** The kernel signature is changing. If there's a stale `.nbi` on a CI runner or dev machine, you get a confusing `TypingError` on first run. Document the "delete __pycache__" step or stop acting surprised when someone files a bug.
4. **Breakeven trigger on `high`.** Now BE activates on the intrabar peak. Fantastic. But what happens on the **same bar** where the intrabar high trips BE *and* the intrabar low trips the freshly-moved SL? Your kernel had better evaluate BE activation *before* the SL check in the same iteration, or you'll get a BE that wasn't-quite-set hammering into the original `lower`.
5. **"All tests pass with `high_prices=close, low_prices=close`."** Translation: "we changed nothing meaningful and shipped." You need **at least one** fixture where high ≠ low ≠ close and assert the label you expect. Otherwise this entire PR is a no-op from the test suite's perspective.
6. **Meta-labeler silent invalidation.** Every cached model trained on the old labels is now toxic. If `tuner.py` or `main_execution.py` lazy-loads a joblib from disk, your "fix" will make *live* decisions against a model trained on fantasy labels. Somebody needs to nuke those artefacts or version-bump the cache key.
7. **Test files using `open_prices=close` as a synthetic open.** Cute hack. Now extend it: those same tests will pass `high_prices=close, low_prices=close` and the PT/SL-via-high/low logic becomes **identical to the old close-only logic** for those cases — which is exactly why the existing tests won't catch the bug fix. You must add **new** tests with genuine OHLC variance.
8. **`main_backtest.py:366` and `main_execution.py:173`.** Two spots. Miss one and you get inconsistent behaviour between the two entry points — one uses real OHLC, the other blows up at runtime. Grep is your friend; don't rely on memory.
9. **Sharpe won't necessarily go up.** The user expects this fix to rescue negative Sharpes. It *might* — but it might also expose that the Alpha was only "winning" because of phantom PT hits that required a close-print confirmation. **Set expectations now** or the user will think the fix is broken.
10. **No explicit versioning of the label schema.** If any downstream artefact (feature store, cached events) was serialised with the old semantics, this PR silently changes interpretation without a version bump. Add a log line on first call documenting the new semantics, or face the wrath of whoever debugs this in production.

### Balanced Response
Grumpy is right on all ten counts; the implementation below bakes in the mitigations:

1. Side-adjusted return convention is restated in the kernel docstring and inline-commented at the `return` statements (§ Proposed Changes / kernel rewrite).
2. "SL-first on ambiguous same-bar PT+SL cross" is both **commented in the kernel** and **explicitly unit-tested** (`test_same_bar_double_hit_prefers_sl`).
3. Numba cache invalidation note added to Verification Plan → Manual Steps.
4. Kernel evaluates BE **activation first**, **then** the (possibly-moved) SL check, inside the same iteration — order is documented in the code and asserted by `test_breakeven_trigger_on_high_then_sl_same_bar`.
5. Test file updates split into two groups: **(a)** existing tests backfilled with `high_prices=close, low_prices=close` to keep them green (preserves their original intent — those tests assert kernel plumbing, not intrabar semantics); **(b)** new test module `tests/test_labeling/test_triple_barrier_ohlc.py` asserts the new intrabar behaviour with deliberately wide bars.
6. User Review Required block lists "retrain/invalidate cached meta-labeler artefacts" as a manual step.
7. New tests explicitly pass `high`/`low` different from `close` — see implementation.
8. Both call sites (`main_backtest.py:366`, `main_execution.py:173`) are updated in the diff below; `grep -n "get_labels(" src/` is in the Verification Plan.
9. User Review Required explicitly flags "Sharpe may move in either direction after the fix; the fix corrects correctness, not profitability."
10. A single `logger.info("Triple Barrier: intrabar High/Low evaluation active (v2 semantics)")` line is emitted once per `apply_triple_barrier` call to aid post-deploy forensics.

## Proposed Changes

> [!IMPORTANT]
> **MAXIMUM DETAIL REQUIRED:** Provide complete, fully functioning code blocks. Break down the logic step-by-step before showing code.

---

### 1. Labeling Kernel — Intrabar High/Low Evaluation

#### MODIFY `src/labeling/triple_barrier.py`

- **Context:** The two `@njit` kernels and the Python wrapper `apply_triple_barrier` all compute exit returns against `close_values[i]` only. This masks PT hits that reverted by close, swallows SL over-runs at the close price, and mis-times the BE activation. The fix threads `high_values` and `low_values` through the call chain and replaces the single `ret` comparison with dual side-adjusted `ret_hi` / `ret_lo` comparisons, returning the exact barrier value on a cross.
- **Logic — side-adjusted intrabar returns:**
  1. For each bar `i` in `[start+1, end]`, compute:
     - `ret_hi = (high_values[i] / entry_price - 1.0) * side`
     - `ret_lo = (low_values[i] / entry_price - 1.0) * side`
     - For `side == +1` (Long): `ret_hi` is the best-case intrabar return, `ret_lo` is the worst-case. So PT check is `ret_hi >= upper`; SL check is `ret_lo <= lower`.
     - For `side == -1` (Short): because the multiplication flips signs, `ret_lo` (raw price down) becomes *positive* side-adjusted (the best case for a short), and `ret_hi` (raw price up) becomes *negative* (the worst case). So for a Short, the "side-adjusted best" is `ret_lo` and "side-adjusted worst" is `ret_hi` — but the comparisons are symmetric: we want the *maximum* side-adjusted return vs. `upper` and the *minimum* side-adjusted return vs. `lower`.
     - Formally: `sa_max = max(ret_hi, ret_lo)`, `sa_min = min(ret_hi, ret_lo)`. PT hit iff `sa_max >= upper`; SL hit iff `sa_min <= lower`. This single formulation is correct for both sides and avoids a branch.
  2. On a cross, return the **barrier value** (`upper` or `lower`) as the realised `ret`, NOT the close-based return. This models a stop/limit fill at the level.
  3. **Ambiguity resolution — SL first:** if both `sa_min <= lower` and `sa_max >= upper` on the same bar, return SL. This is the LdP convention and the conservative fill assumption (we cannot know intrabar path ordering from bar data alone).
  4. On vertical expiry (`end` reached without a cross), preserve the existing close-based `final_ret = (close_values[end] / entry_price - 1.0) * side`.
- **Logic — Breakeven (`_find_dynamic_touch`):**
  1. BE activation check uses `sa_max` (the best-case intrabar side-adjusted return) against `upper * be_trigger`.
  2. On activation, set `lower = 0.0001` (unchanged semantics).
  3. **Critical ordering within a single bar iteration:** BE activation check runs *first*, THEN PT check, THEN SL check (against possibly-moved `lower`). This preserves the "BE activated this bar then SL hit same bar" path correctly.
- **Logic — Wrapper (`apply_triple_barrier`):**
  1. Add `high_prices: pd.Series` and `low_prices: pd.Series` as required kwargs (same enforcement style as existing `open_prices`).
  2. Validate: raise `AssertionError` if either is `None`, mirroring the existing `open_prices` guard.
  3. Convert both to `np.float64` arrays once before the loop (same as `close_values`).
  4. Pass through to `_find_dynamic_touch`.
  5. Emit one-shot `logger.info` on entry announcing v2 semantics (aids post-deploy forensics per Grumpy §10).
- **Implementation (complete rewrite of `src/labeling/triple_barrier.py`):**

```python
"""
Método da Tripla Barreira — TradeSystem5000.

Este módulo implementa a técnica de rotulagem de Tripla Barreira, que avalia
o resultado de um sinal através de três limites simultâneos:
1.  **Barreira Superior (Profit Take)**: Meta de lucro proporcional à volatilidade.
2.  **Barreira Inferior (Stop Loss)**: Limite de perda com suporte a breakeven.
3.  **Barreira Vertical (Time Stop)**: Tempo máximo de permanência na posição.

Semântica v2 (2026-04-13): as barreiras de PT e SL são avaliadas contra a
**Máxima e Mínima intrabar** (não contra o fechamento). Quando há cruzamento,
retornamos o **valor cravado da barreira** (`upper`/`lower`), simulando uma
ordem limit/stop preenchida no nível. Apenas a barreira vertical (time stop)
continua a usar o fechamento da barra terminal (`final_ret`).

Convenção de sinal: todos os retornos são armazenados em forma *side-adjusted*
(`ret * side`). Para Long (`side=+1`) coincide com o retorno bruto de preço;
para Short (`side=-1`) é o negado. Todas as comparações com `upper`/`lower`
operam em valores side-adjusted.

Ambiguidade intrabar: se uma única barra cruzar simultaneamente `upper` e
`lower` (Max >= PT e Min <= SL), retornamos **SL** — convenção conservadora
de López de Prado.

Funcionalidades:
- **apply_triple_barrier**: Aplicação dos limites e detecção do primeiro toque.
- **create_events**: Preparação estruturada dos eventos de entrada.
- **get_labels**: Atribuição de classes {-1, 0, 1} para o Alpha.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulo 3.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from numba import njit

from config.settings import labeling_config


# ---------------------------------------------------------------------------
# Criação de eventos (DataFrame de input para a tripla barreira)
# ---------------------------------------------------------------------------
def create_events(
    close: pd.Series,
    event_timestamps: pd.DatetimeIndex,
    target_vol: pd.Series,
    side: pd.Series | None = None,
    pt_sl: tuple[float, float] | None = None,
    max_holding: int | None = None,
) -> pd.DataFrame:
    """
    Monta o DataFrame de eventos para aplicação da Tripla Barreira.

    Parameters
    ----------
    close : pd.Series
        Série de preços de fechamento (completa).
    event_timestamps : pd.DatetimeIndex
        Timestamps onde há sinal do Alpha (momentos de entrada).
    target_vol : pd.Series
        Volatilidade estimada nos eventos (define largura das barreiras).
    side : pd.Series, optional
        Direção do sinal (+1 compra, -1 venda). Se None, assume bidirecional.
    pt_sl : tuple[float, float], optional
        Multiplicadores (take_profit, stop_loss) aplicados à volatilidade.
        Default: config.
    max_holding : int, optional
        Número máximo de barras na posição (barreira vertical).
        Default: config.

    Returns
    -------
    pd.DataFrame
        Colunas: ``t1`` (barreira vertical), ``trgt`` (volatilidade alvo),
        ``side`` (direção).
    """
    if pt_sl is None:
        pt_sl = labeling_config.pt_sl_ratio
    if max_holding is None:
        max_holding = labeling_config.max_holding_periods

    common = event_timestamps.intersection(close.index).intersection(target_vol.index)
    if len(common) < len(event_timestamps):
        logger.debug(
            "Eventos filtrados: {} → {} (interseção com close+vol)",
            len(event_timestamps),
            len(common),
        )

    events = pd.DataFrame(index=common)

    close_index = close.index
    t1_values = []
    for ts in events.index:
        loc = close_index.get_loc(ts)
        end_loc = min(loc + max_holding, len(close_index) - 1)
        t1_values.append(close_index[end_loc])
    events["t1"] = pd.DatetimeIndex(t1_values)

    events["trgt"] = target_vol.reindex(events.index)

    if side is not None:
        events["side"] = side.reindex(events.index)
    else:
        events["side"] = 1

    events = events.dropna(subset=["trgt"])

    events.attrs["pt_sl"] = pt_sl

    logger.info(
        "Eventos criados: {} | pt_sl={} | max_holding={} barras",
        len(events),
        pt_sl,
        max_holding,
    )
    return events


# ---------------------------------------------------------------------------
# Aplicação da Tripla Barreira
# ---------------------------------------------------------------------------
def apply_triple_barrier(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float] | None = None,
    be_trigger: float = 0.0,
    open_prices: pd.Series | None = None,
    high_prices: pd.Series | None = None,
    low_prices: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Aplica o método da Tripla Barreira (semântica v2 — intrabar High/Low).

    Para cada evento, verifica qual barreira é atingida primeiro:
    - Superior (PT): side-adjusted max(ret_hi, ret_lo) >= pt * trgt
      → retorno retornado = `upper` (valor cravado da barreira)
    - Inferior (SL): side-adjusted min(ret_hi, ret_lo) <= -sl * trgt
      → retorno retornado = `lower` (valor cravado da barreira)
    - Vertical: tempo atingiu t1
      → retorno retornado = (close[end] / entry - 1) * side

    Ambiguidade (Max e Min cruzam no mesmo candle): retorna SL.

    Parameters
    ----------
    close : pd.Series
        Série completa de preços de fechamento.
    events : pd.DataFrame
        Output de ``create_events``. Colunas: ``t1``, ``trgt``, ``side``.
    pt_sl : tuple[float, float], optional
        Multiplicadores (profit_take, stop_loss). Se 0, desativa a barreira.
        Default: usa o valor armazenado em ``events.attrs``.
    be_trigger : float, optional
        Gatilho para breakeven (fração do Take Profit). Ex: 0.5.
    open_prices : pd.Series
        **Obrigatório.** Preços de abertura (para entry_price em T+1).
    high_prices : pd.Series
        **Obrigatório.** Máximas intrabar (avaliação de PT/SL).
    low_prices : pd.Series
        **Obrigatório.** Mínimas intrabar (avaliação de PT/SL).

    Returns
    -------
    pd.DataFrame
        Colunas: ``t1``, ``ret`` (side-adjusted), ``trgt``, ``side``, ``barrier_type``.
    """
    if pt_sl is None:
        pt_sl = events.attrs.get("pt_sl", labeling_config.pt_sl_ratio)

    pt_mult, sl_mult = pt_sl

    if open_prices is None:
        raise AssertionError(
            "triple_barrier: open_prices é obrigatório. O fallback para "
            "close_values[start_loc] introduz lookahead (entrada no fechamento "
            "da mesma barra que gerou o sinal). Passe a série de aberturas."
        )
    if high_prices is None or low_prices is None:
        raise AssertionError(
            "triple_barrier: high_prices e low_prices são obrigatórios (semântica v2). "
            "Para preservar o comportamento antigo em testes sintéticos, passe "
            "high_prices=close e low_prices=close explicitamente."
        )

    logger.info("Triple Barrier: intrabar High/Low evaluation active (v2 semantics)")

    results = []
    close_values = close.values.astype(np.float64)
    high_values = high_prices.values.astype(np.float64)
    low_values = low_prices.values.astype(np.float64)
    close_idx = close.index

    for event_ts, row in events.iterrows():
        t1 = row["t1"]
        trgt = row["trgt"]
        side = row["side"]

        try:
            start_loc = close_idx.get_loc(event_ts)
        except KeyError:
            continue

        if pd.isna(t1):
            end_loc = len(close_idx) - 1
        else:
            try:
                end_loc = close_idx.get_loc(t1)
            except KeyError:
                end_loc = close_idx.searchsorted(t1, side="right") - 1

        if start_loc + 1 >= len(close_values) or end_loc <= start_loc:
            continue

        entry_price = open_prices.values[start_loc + 1]

        upper = trgt * pt_mult if pt_mult > 0 else np.inf
        lower = -trgt * sl_mult if sl_mult > 0 else -np.inf

        touch_ts, touch_ret, barrier_type = _find_dynamic_touch(
            close_values=close_values,
            high_values=high_values,
            low_values=low_values,
            start=start_loc,
            end=end_loc,
            entry_price=entry_price,
            side=int(side),
            upper=upper,
            lower=lower,
            be_trigger=be_trigger,
        )

        if touch_ts < len(close_idx):
            result_ts = close_idx[touch_ts]
        else:
            result_ts = close_idx[-1]

        results.append({
            "event_ts": event_ts,
            "t1": result_ts,
            "ret": touch_ret,
            "trgt": trgt,
            "side": side,
            "barrier_type": barrier_type,
        })

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df.set_index("event_ts", inplace=True)
        result_df.index.name = None

    if len(result_df) > 0:
        counts = result_df["barrier_type"].value_counts()
        logger.info(
            "Tripla Barreira: {} eventos | pt={}, sl={}, vertical={}",
            len(result_df),
            counts.get(0, 0),  # pt
            counts.get(1, 0),  # sl
            counts.get(2, 0),  # vertical
        )
    else:
        logger.warning("Tripla Barreira: nenhum evento processado")

    return result_df


# ---------------------------------------------------------------------------
# Kernel otimizado (semântica v2 — intrabar High/Low)
# ---------------------------------------------------------------------------
@njit
def _find_first_touch(
    close_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    start: int,
    end: int,
    entry_price: float,
    side: int,
    upper: float,
    lower: float,
) -> tuple[int, float, int]:
    """
    Versão sem breakeven — preservada para compatibilidade / benchmarking.

    Semântica v2: avalia PT/SL contra os extremos intrabar; na ambiguidade
    (mesma barra cruza ambas), retorna SL (conservador).

    Returns
    -------
    (index_do_toque, retorno_side_adjusted, tipo_barreira)
    tipo_barreira: 0=pt, 1=sl, 2=vertical
    """
    for i in range(start + 1, end + 1):
        ret_hi = (high_values[i] / entry_price - 1.0) * side
        ret_lo = (low_values[i] / entry_price - 1.0) * side

        # Extremos side-adjusted (válidos para Long e Short)
        sa_max = ret_hi if ret_hi > ret_lo else ret_lo
        sa_min = ret_hi if ret_hi < ret_lo else ret_lo

        sl_hit = sa_min <= lower
        pt_hit = sa_max >= upper

        # Ambiguidade: SL vence (convenção LdP, fill conservador)
        if sl_hit:
            return i, lower, 1  # sl — retorna valor cravado da barreira
        if pt_hit:
            return i, upper, 0  # pt — retorna valor cravado da barreira

    # Barreira vertical: retorno do fechamento na última barra
    final_ret = (close_values[end] / entry_price - 1.0) * side
    return end, final_ret, 2  # vertical


@njit
def _find_dynamic_touch(
    close_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    start: int,
    end: int,
    entry_price: float,
    side: int,
    upper: float,
    lower: float,
    be_trigger: float = 0.0,
) -> tuple[int, float, int]:
    """
    Versão com Breakeven — semântica v2.

    Ordem de avaliação dentro de cada barra (crítico):
      1. Ativação de BE (se `be_trigger > 0` e sa_max atingiu upper*be_trigger).
      2. Checagem de PT (sa_max >= upper).
      3. Checagem de SL (sa_min <= lower — contra o lower eventualmente movido).

    Ambiguidade (Max e Min cruzam no mesmo candle): SL vence.

    Returns
    -------
    (index_do_toque, retorno_side_adjusted, tipo_barreira)
    tipo_barreira: 0=pt, 1=sl, 2=vertical
    """
    breakeven_active = False

    for i in range(start + 1, end + 1):
        ret_hi = (high_values[i] / entry_price - 1.0) * side
        ret_lo = (low_values[i] / entry_price - 1.0) * side

        sa_max = ret_hi if ret_hi > ret_lo else ret_lo
        sa_min = ret_hi if ret_hi < ret_lo else ret_lo

        # 1. Ativação de BE ANTES dos checks de barreira, para capturar
        #    o caso "BE ativou e SL foi atingido na mesma barra".
        if not breakeven_active and be_trigger > 0 and sa_max >= (upper * be_trigger):
            lower = 0.0001
            breakeven_active = True

        sl_hit = sa_min <= lower
        pt_hit = sa_max >= upper

        # 2/3. Ambiguidade intrabar: SL vence (fill conservador).
        if sl_hit:
            return i, lower, 1  # sl — valor cravado
        if pt_hit:
            return i, upper, 0  # pt — valor cravado

    final_ret = (close_values[end] / entry_price - 1.0) * side
    return end, final_ret, 2  # vertical


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BARRIER_NAMES = {0: "pt", 1: "sl", 2: "vertical"}


def get_labels(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float] | None = None,
    be_trigger: float = 0.0,
    min_return: float | None = None,
    open_prices: pd.Series | None = None,
    high_prices: pd.Series | None = None,
    low_prices: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Pipeline completo: aplica tripla barreira e gera labels.

    Labels:
    - +1: Take Profit atingido (trade lucrativo)
    - -1: Stop Loss atingido (trade perdedor)
    - 0: Barreira vertical (tempo esgotado, retorno insuficiente)

    Ver `apply_triple_barrier` para a semântica v2 (intrabar High/Low).

    Parameters
    ----------
    close : pd.Series
        Preços de fechamento.
    events : pd.DataFrame
        Eventos com t1, trgt, side.
    pt_sl : tuple, optional
        Multiplicadores PT/SL.
    be_trigger : float, optional
        Gatilho para breakeven.
    min_return : float, optional
        Retorno mínimo para considerar label != 0.
    open_prices : pd.Series
        **Obrigatório.**
    high_prices : pd.Series
        **Obrigatório (semântica v2).**
    low_prices : pd.Series
        **Obrigatório (semântica v2).**

    Returns
    -------
    pd.DataFrame
        Colunas: ``t1``, ``ret``, ``label``, ``side``, ``barrier_type``.
    """
    if min_return is None:
        min_return = labeling_config.min_return

    # Curto-circuito para inputs vazios (preservado do comportamento anterior)
    if len(close) == 0 or len(events) == 0:
        return pd.DataFrame()

    result = apply_triple_barrier(
        close, events, pt_sl,
        be_trigger=be_trigger,
        open_prices=open_prices,
        high_prices=high_prices,
        low_prices=low_prices,
    )

    if len(result) == 0:
        return result

    result["barrier_type"] = result["barrier_type"].map(_BARRIER_NAMES)

    result["label"] = 0
    result.loc[result["barrier_type"] == "pt", "label"] = 1
    result.loc[result["barrier_type"] == "sl", "label"] = -1

    vertical_mask = result["barrier_type"] == "vertical"
    result.loc[vertical_mask & (result["ret"] > min_return), "label"] = 1
    result.loc[vertical_mask & (result["ret"] < -min_return), "label"] = -1

    counts = result["label"].value_counts()
    logger.success(
        "Labels: +1={}, -1={}, 0={}",
        counts.get(1, 0),
        counts.get(-1, 0),
        counts.get(0, 0),
    )
    return result
```

- **Edge Cases Handled:**
  - `sl_hit` evaluated before `pt_hit` inside the same bar → SL-first ambiguity convention.
  - BE activation runs *before* PT/SL checks inside the same bar iteration → "BE activated then SL hit same bar" is deterministic.
  - Side-symmetric `sa_max`/`sa_min` derivation works for both Long and Short without branching.
  - Required `high_prices`/`low_prices` prevents silent reversion to close-only semantics.
  - Existing guard for `open_prices` preserved.
  - `get_labels` short-circuit for empty inputs preserved (avoids passing empty series into validation).

---

### 2. Backtest Entry Point — Pass OHLC into Labeling

#### MODIFY `src/main_backtest.py`

- **Context:** The backtest already reads full OHLC bars into `df`. The `get_labels` call at line 366 currently passes only `close` and `open`; we must add `high`/`low`.
- **Logic:** single-line keyword-argument additions to the existing call.
- **Implementation (surgical diff — line 366 area):**

```python
# BEFORE:
    labels_df = get_labels(df["close"], events, be_trigger=be_trigger, open_prices=df["open"])

# AFTER:
    labels_df = get_labels(
        df["close"],
        events,
        be_trigger=be_trigger,
        open_prices=df["open"],
        high_prices=df["high"],
        low_prices=df["low"],
    )
```

- **Edge Cases Handled:** `df` is guaranteed to have `high`/`low` columns by the upstream data-loader (same schema that provides `open`). No new NaN handling beyond what `apply_triple_barrier` already tolerates.

---

### 3. Live/Paper Execution Entry Point — Pass OHLC into Labeling

#### MODIFY `src/main_execution.py`

- **Context:** Mirror of §2 for the live/paper pipeline. `df_aligned` carries the same OHLC schema.
- **Logic:** identical kwarg additions.
- **Implementation (surgical diff — line 173 area):**

```python
# BEFORE:
    labels_df = get_labels(df_aligned["close"], events, open_prices=df_aligned["open"])

# AFTER:
    labels_df = get_labels(
        df_aligned["close"],
        events,
        open_prices=df_aligned["open"],
        high_prices=df_aligned["high"],
        low_prices=df_aligned["low"],
    )
```

- **Edge Cases Handled:** same as §2.

---

### 4. Existing Tests — Keep Green with Synthetic OHLC

#### MODIFY `tests/test_labeling/test_triple_barrier.py`

- **Context:** Three call sites pass only `open_prices=close` (synthetic). To preserve the exact pre-fix behaviour for these tests (they assert kernel plumbing, not intrabar semantics), pass `high_prices=close, low_prices=close` — when High=Low=Close, `sa_max == sa_min == ret_close`, so the kernel collapses to the original single-return comparison and all existing assertions hold.
- **Logic:** Add `high_prices=close, low_prices=close` to every `apply_triple_barrier(...)` and `get_labels(...)` call.
- **Implementation:**

```python
# Line 37:
    result = apply_triple_barrier(close, events, open_prices=close, high_prices=close, low_prices=close)

# Line 53:
    result = apply_triple_barrier(close, events, be_trigger=0.5, open_prices=close, high_prices=close, low_prices=close)

# Line 60:
    labels = get_labels(close, events, open_prices=close, high_prices=close, low_prices=close)

# Line 67 (empty inputs test):
    labels = get_labels(close, events)   # short-circuit handles empty inputs; no OHLC needed
```

- **Edge Cases Handled:** the empty-inputs test at line 67 relies on the early-return in `get_labels` for `len(close) == 0 or len(events) == 0`, which is preserved above.

---

### 5. Phase-3 Tests — Keep Green

#### MODIFY `tests/test_labeling/test_phase3.py`

- **Context:** Five call sites (lines 216, 234, 250, 273, 291, 318) use `open_prices=close`. Same synthetic-OHLC treatment.
- **Logic:** append `high_prices=close, low_prices=close` to each call.
- **Implementation (pattern — apply to each line):**

```python
# BEFORE (example line 216):
        labels = get_labels(close, events, pt_sl=(0.5, 0.5), open_prices=close)

# AFTER:
        labels = get_labels(close, events, pt_sl=(0.5, 0.5), open_prices=close, high_prices=close, low_prices=close)
```

Apply the same edit to lines 234, 250, 273, 291, 318 (the `get_labels(...)` and `apply_triple_barrier(...)` calls identified by `grep -n "(open_prices=close)" tests/test_labeling/test_phase3.py`).

- **Edge Cases Handled:** synthetic OHLC (`high=low=close`) collapses v2 semantics to the pre-fix behaviour — all phase-3 assertions remain valid.

---

### 6. Dynamic Barrier Tests — Keep Green

#### MODIFY `tests/test_labeling/test_dynamic_barrier.py`

- **Context:** Four `get_labels(...)` call sites at lines 43, 48, 69, 90 use `open_prices=close` (or `open_prices=open_prices`). Apply the same synthetic-OHLC pattern.
- **Logic:** append `high_prices=close, low_prices=close` where `close` is the variable name in scope (or `high_prices=open_prices, low_prices=open_prices` if the test constructed a distinct open series — double-check the local variable name before editing each line).
- **Implementation (pattern):**

```python
# BEFORE (line 43):
    labels_no_be = get_labels(close, events, pt_sl=(1.0, 1.0), be_trigger=0.0, open_prices=open_prices)

# AFTER:
    labels_no_be = get_labels(close, events, pt_sl=(1.0, 1.0), be_trigger=0.0, open_prices=open_prices, high_prices=close, low_prices=close)
```

Repeat for lines 48, 69, 90.

- **Edge Cases Handled:** `high=low=close` collapses v2 to the pre-fix semantics — existing BE assertions (`barrier_type == "sl"`, returns near zero) remain valid.

---

### 7. New OHLC Semantics Tests

#### CREATE `tests/test_labeling/test_triple_barrier_ohlc.py`

- **Context:** No existing test exercises the new intrabar behaviour. We need dedicated fixtures where `high != low != close` to assert the four distinct outcomes: (a) PT hit by high only, close reverts; (b) SL hit by low only, close recovers; (c) same-bar PT+SL → SL wins; (d) BE activates on intrabar high then SL hits same bar.
- **Logic:**
  1. Construct a minimal pd.Series for close/open/high/low over ~15 bars.
  2. Seed one event at bar 0; `t1` at bar 14.
  3. Craft specific bars to force each scenario and assert `barrier_type` and `ret`.
- **Implementation (full file):**

```python
"""
Testes para a semântica v2 da Tripla Barreira (avaliação intrabar High/Low).

Cobre os quatro cenários distintos que a versão close-only não conseguia
detectar corretamente:
  1. PT tocado pela Máxima, fechamento reverte.
  2. SL tocado pela Mínima, fechamento recupera.
  3. Ambiguidade same-bar (PT e SL cruzados) → SL vence.
  4. BE ativado pela Máxima e SL atingido na mesma barra.
"""
import numpy as np
import pandas as pd
import pytest

from src.labeling.triple_barrier import (
    apply_triple_barrier,
    create_events,
    get_labels,
)


def _build_ohlc(bars: list[dict]) -> dict[str, pd.Series]:
    """Helper: converte lista de dicts OHLC em Series indexadas por timestamp."""
    idx = pd.date_range("2026-01-01 10:00", periods=len(bars), freq="1min")
    df = pd.DataFrame(bars, index=idx)
    return {
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["close"],
    }


def test_pt_hit_by_high_close_reverts():
    """
    Cenário A: Long. Entry @ bar1 open = 100. PT = +2%, SL = -2%.
    Bar 3: high 102.5 (cruza PT), close 100.5 (reverte).
    Semântica v1 (buggy): não detecta — continua até vertical.
    Semântica v2: detecta PT na bar 3, retorna ret = +0.02.
    """
    bars = [
        {"open": 100, "high": 100.1, "low": 99.9, "close": 100},
        {"open": 100, "high": 100.5, "low": 99.9, "close": 100.2},  # bar1 (entry)
        {"open": 100.2, "high": 100.8, "low": 100.0, "close": 100.3},
        {"open": 100.3, "high": 102.5, "low": 100.2, "close": 100.5},  # PT crossed by high
        {"open": 100.5, "high": 100.7, "low": 100.3, "close": 100.4},
    ] + [{"open": 100.4, "high": 100.5, "low": 100.3, "close": 100.4}] * 5

    ohlc = _build_ohlc(bars)
    close = ohlc["close"]

    events = pd.DataFrame({
        "t1": [close.index[9]],
        "trgt": [0.02],
        "side": [1],
    }, index=[close.index[0]])

    result = apply_triple_barrier(
        close, events, pt_sl=(1.0, 1.0),
        open_prices=ohlc["open"],
        high_prices=ohlc["high"],
        low_prices=ohlc["low"],
    )

    assert len(result) == 1
    assert result.iloc[0]["barrier_type"] == 0  # pt
    assert result.iloc[0]["ret"] == pytest.approx(0.02, abs=1e-9)


def test_sl_hit_by_low_close_recovers():
    """
    Cenário B: Long. Bar 3: low 97.5 (cruza SL), close 100 (recupera).
    Semântica v2: SL na bar 3, ret = -0.02.
    """
    bars = [
        {"open": 100, "high": 100.1, "low": 99.9, "close": 100},
        {"open": 100, "high": 100.3, "low": 99.8, "close": 100.1},  # bar1 (entry)
        {"open": 100.1, "high": 100.2, "low": 99.7, "close": 99.9},
        {"open": 99.9, "high": 100.0, "low": 97.5, "close": 100.0},  # SL crossed by low
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0},
    ] + [{"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0}] * 5

    ohlc = _build_ohlc(bars)
    close = ohlc["close"]

    events = pd.DataFrame({
        "t1": [close.index[9]],
        "trgt": [0.02],
        "side": [1],
    }, index=[close.index[0]])

    result = apply_triple_barrier(
        close, events, pt_sl=(1.0, 1.0),
        open_prices=ohlc["open"],
        high_prices=ohlc["high"],
        low_prices=ohlc["low"],
    )

    assert len(result) == 1
    assert result.iloc[0]["barrier_type"] == 1  # sl
    assert result.iloc[0]["ret"] == pytest.approx(-0.02, abs=1e-9)


def test_same_bar_double_hit_prefers_sl():
    """
    Cenário C: bar cruza ambos → SL vence (convenção LdP, fill conservador).
    """
    bars = [
        {"open": 100, "high": 100.1, "low": 99.9, "close": 100},
        {"open": 100, "high": 100.2, "low": 99.9, "close": 100.0},  # bar1 (entry)
        {"open": 100.0, "high": 102.5, "low": 97.5, "close": 100.0},  # both crossed
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0},
    ] + [{"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0}] * 6

    ohlc = _build_ohlc(bars)
    close = ohlc["close"]

    events = pd.DataFrame({
        "t1": [close.index[9]],
        "trgt": [0.02],
        "side": [1],
    }, index=[close.index[0]])

    result = apply_triple_barrier(
        close, events, pt_sl=(1.0, 1.0),
        open_prices=ohlc["open"],
        high_prices=ohlc["high"],
        low_prices=ohlc["low"],
    )

    assert len(result) == 1
    assert result.iloc[0]["barrier_type"] == 1  # SL wins on ambiguity
    assert result.iloc[0]["ret"] == pytest.approx(-0.02, abs=1e-9)


def test_breakeven_trigger_on_high_then_sl_same_bar():
    """
    Cenário D: BE ativa na high (>= upper*be_trigger=0.01) e SL movido (0.0001)
    é atingido pela low na mesma barra. Esperado: barrier_type=sl, ret≈0.0001.
    """
    # Entry bar1 open=100. upper=0.02 (PT @ 2%). be_trigger=0.5 → ativa em +1%.
    # Bar 3: high=101.5 (ativa BE), low=99.5 (<= 0.0001? 99.5/100-1=-0.005, sim).
    bars = [
        {"open": 100, "high": 100.1, "low": 99.9, "close": 100},
        {"open": 100, "high": 100.3, "low": 99.9, "close": 100.2},  # bar1 (entry)
        {"open": 100.2, "high": 100.5, "low": 100.0, "close": 100.3},
        {"open": 100.3, "high": 101.5, "low": 99.5, "close": 100.0},  # BE on, then SL
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0},
    ] + [{"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0}] * 5

    ohlc = _build_ohlc(bars)
    close = ohlc["close"]

    events = pd.DataFrame({
        "t1": [close.index[9]],
        "trgt": [0.02],
        "side": [1],
    }, index=[close.index[0]])

    result = apply_triple_barrier(
        close, events, pt_sl=(1.0, 1.0),
        be_trigger=0.5,
        open_prices=ohlc["open"],
        high_prices=ohlc["high"],
        low_prices=ohlc["low"],
    )

    assert len(result) == 1
    assert result.iloc[0]["barrier_type"] == 1  # sl (post-BE)
    assert result.iloc[0]["ret"] == pytest.approx(0.0001, abs=1e-9)


def test_short_side_pt_hit_by_low():
    """
    Short: Entry @ 100. PT = +2% side-adjusted → preço cai para 98.
    Bar 3: low 97.5 (cruza PT para short), close 100 (reverte).
    Semântica v2: detecta PT, ret = +0.02 (side-adjusted).
    """
    bars = [
        {"open": 100, "high": 100.1, "low": 99.9, "close": 100},
        {"open": 100, "high": 100.3, "low": 99.7, "close": 100.0},  # bar1 (entry short)
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0},
        {"open": 100.0, "high": 100.1, "low": 97.5, "close": 100.0},  # short PT by low
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0},
    ] + [{"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0}] * 5

    ohlc = _build_ohlc(bars)
    close = ohlc["close"]

    events = pd.DataFrame({
        "t1": [close.index[9]],
        "trgt": [0.02],
        "side": [-1],
    }, index=[close.index[0]])

    result = apply_triple_barrier(
        close, events, pt_sl=(1.0, 1.0),
        open_prices=ohlc["open"],
        high_prices=ohlc["high"],
        low_prices=ohlc["low"],
    )

    assert len(result) == 1
    assert result.iloc[0]["barrier_type"] == 0  # pt
    assert result.iloc[0]["ret"] == pytest.approx(0.02, abs=1e-9)
```

- **Edge Cases Handled:** covers all four ambiguity/timing scenarios raised by Grumpy; includes Short-side validation (side-symmetry of `sa_max`/`sa_min`).

## Verification Plan

### Automated Tests
- **Existing tests — must all remain green:**
  - `tests/test_labeling/test_triple_barrier.py` (4 tests after edits).
  - `tests/test_labeling/test_phase3.py` (all existing cases, edited to pass synthetic OHLC).
  - `tests/test_labeling/test_dynamic_barrier.py` (all existing cases).
  - `tests/test_main_execution.py` — validate the top-level pipeline still runs end-to-end (it imports `get_labels` through `src/main_execution.py`).
- **New tests — must all pass:**
  - `tests/test_labeling/test_triple_barrier_ohlc.py::test_pt_hit_by_high_close_reverts`
  - `tests/test_labeling/test_triple_barrier_ohlc.py::test_sl_hit_by_low_close_recovers`
  - `tests/test_labeling/test_triple_barrier_ohlc.py::test_same_bar_double_hit_prefers_sl`
  - `tests/test_labeling/test_triple_barrier_ohlc.py::test_breakeven_trigger_on_high_then_sl_same_bar`
  - `tests/test_labeling/test_triple_barrier_ohlc.py::test_short_side_pt_hit_by_low`
- **Run commands:**
  ```bash
  # Fast: labeling only
  pytest tests/test_labeling/ -v

  # Full regression (includes execution + main_execution integration)
  pytest tests/ -v
  ```
- **Grep guardrails (pre-merge):**
  ```bash
  # Ensure no stale close-only call sites remain
  grep -rn "apply_triple_barrier(" src/ tests/
  grep -rn "get_labels("          src/ tests/
  # Every hit must pass both high_prices= and low_prices=
  ```

### Manual / Operational Steps
- **Clear Numba cache** on first deploy to avoid stale-signature `TypingError`:
  ```bash
  find . -name "*.nbi" -delete
  find . -name "*.nbc" -delete
  find . -type d -name "__pycache__" -exec rm -rf {} +
  ```
- **Invalidate cached meta-labeler artefacts** trained on pre-fix labels (joblib/pkl files produced by prior `main_backtest.py` or `tuner.py` runs). Re-train before relying on new Sharpe figures.
- **Re-run Optuna study** — prior `tuner.py` trials penalised the Alpha based on buggy labels; delete or archive the old study database and start fresh.
- **Sanity check on real data:** run one full historical backtest on a representative instrument (e.g. WINFUT 5-minute bars, full last-quarter window) both before and after, capturing:
  - Label distribution (`labels_df["label"].value_counts()`).
  - PT/SL/vertical split (`result["barrier_type"].value_counts()`).
  - Mean/median `ret` per label class.
  - Final equity curve and Sharpe.
  Expect: PT and SL share **increases**, vertical share **decreases**; mean |ret| on SL hits moves **toward** the exact `lower` value (previously dispersed by close over-run).
- **Expectation management:** the Sharpe may move in either direction. The fix corrects *correctness*, not profitability. If Sharpe stays negative after retraining the meta-labeler, the Alpha itself needs review.

---

## Recommended Agent
**Send to Lead Coder** — the Numba kernel rewrite, the side-adjusted symmetry derivation, and the silent-behaviour-change risk across cached models all qualify this as Complex/Risky per the pair-programming optimisation rules.

---

## Review Results (2026-04-13)

### Status: READY

### Files Changed
- `src/labeling/triple_barrier.py` — full v2 rewrite (intrabar High/Low); removed `_find_first_touch` (dead Numba code, never called).
- `src/main_backtest.py` — `get_labels` call updated with `high_prices=df["high"]`, `low_prices=df["low"]`.
- `src/main_execution.py` — `get_labels` call updated with `high_prices=df_aligned["high"]`, `low_prices=df_aligned["low"]`.
- `tests/test_labeling/test_triple_barrier.py` — all calls updated with synthetic OHLC; unused `_find_dynamic_touch` import removed; explanatory comment added to `test_get_labels_empty`.
- `tests/test_labeling/test_phase3.py` — all `get_labels`/`apply_triple_barrier` calls updated with `high_prices=close, low_prices=close`.
- `tests/test_labeling/test_dynamic_barrier.py` — all `get_labels` calls updated with synthetic OHLC.
- `tests/test_labeling/test_triple_barrier_ohlc.py` — **new file**: 5 tests exercising genuine intrabar OHLC behaviour.

### Reviewer Findings

| Severity | Finding | Resolution |
|----------|---------|------------|
| MAJOR | `_find_first_touch` was orphaned dead code compiled by Numba — zero callers in production or tests | **Fixed**: removed the function. Recoverable from git history if benchmarking is ever needed. |
| MAJOR | `test_get_labels_empty` called `get_labels` without OHLC args — relied silently on short-circuit ordering with no documentation | **Fixed**: added an explicit comment explaining the short-circuit dependency and the expected failure mode if it changes. |
| NIT | BE trigger with `upper=inf` (PT disabled): `sa_max >= inf` always False — BE never activates. Correct behaviour, undocumented. | Deferred — low operational risk, no new callers affected. |
| NIT | `be_trigger: float = 0.0` default in `@njit` — always passed explicitly, default is unused cargo-cult | Deferred — no functional impact. |

### Validation Results
```
300 passed in 8.16s
  - tests/test_labeling/test_triple_barrier_ohlc.py: 5/5 new OHLC tests PASSED
  - tests/test_labeling/: 41/41 PASSED (all existing + new)
  - tests/ (full suite): 300/300 PASSED
```

### Remaining Risks (Manual — Not Addressable by Tests)
- **Numba cache**: delete `__pycache__/*.nbi` / `*.nbc` on first deploy to avoid stale-signature `TypingError`.
- **Cached meta-labeler artefacts**: joblib/pkl models trained on pre-fix labels are now untrustworthy. Retrain before relying on new Sharpe figures.
- **Optuna study**: prior trials penalised the Alpha on buggy labels; delete/archive the old study database and restart.
- **Sharpe expectation**: the fix corrects correctness, not profitability. Sharpe may move in either direction after retraining.

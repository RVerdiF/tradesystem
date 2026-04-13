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
        Colunas: ``t1`` (timestamp da barreira), ``ret`` (side-adjusted),
        ``trgt``, ``side``, ``barrier_type``.
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

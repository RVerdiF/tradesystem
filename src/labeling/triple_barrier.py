"""
3.3 — Método da Tripla Barreira.

Implementa a rotulagem baseada em três barreiras simultâneas:
- **Superior (Take Profit)**: preço atinge lucro alvo
- **Inferior (Stop Loss)**: preço atinge perda máxima
- **Vertical (Tempo)**: tempo máximo de permanência esgotado

Cada observação recebe o label da primeira barreira tocada.

Referência: López de Prado, *Advances in Financial Machine Learning*, Cap. 3.4.
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

    # Filtra eventos que existem tanto no close quanto no target_vol
    common = event_timestamps.intersection(close.index).intersection(target_vol.index)
    if len(common) < len(event_timestamps):
        logger.debug(
            "Eventos filtrados: {} → {} (interseção com close+vol)",
            len(event_timestamps),
            len(common),
        )

    events = pd.DataFrame(index=common)

    # Barreira vertical: timestamp máximo de permanência
    close_index = close.index
    t1_values = []
    for ts in events.index:
        loc = close_index.get_loc(ts)
        end_loc = min(loc + max_holding, len(close_index) - 1)
        t1_values.append(close_index[end_loc])
    events["t1"] = pd.DatetimeIndex(t1_values)

    # Target de volatilidade
    events["trgt"] = target_vol.reindex(events.index)

    # Direção do sinal
    if side is not None:
        events["side"] = side.reindex(events.index)
    else:
        events["side"] = 1  # assume long por padrão

    # Remove eventos sem target
    events = events.dropna(subset=["trgt"])

    # Armazena pt_sl para uso posterior
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
) -> pd.DataFrame:
    """
    Aplica o método da Tripla Barreira e retorna a primeira barreira tocada.

    Para cada evento, verifica qual barreira é atingida primeiro:
    - Superior: retorno ≥ ``pt * trgt``
    - Inferior: retorno ≤ ``-sl * trgt``
    - Vertical: tempo atingiu ``t1``

    Parameters
    ----------
    close : pd.Series
        Série completa de preços de fechamento.
    events : pd.DataFrame
        Output de ``create_events``. Colunas: ``t1``, ``trgt``, ``side``.
    pt_sl : tuple[float, float], optional
        Multiplicadores (profit_take, stop_loss). Se 0, desativa a barreira.
        Default: usa o valor armazenado em ``events.attrs``.

    Returns
    -------
    pd.DataFrame
        Colunas:
        - ``t1``: timestamp da primeira barreira tocada
        - ``ret``: retorno no momento do toque
        - ``trgt``: volatilidade alvo usada
        - ``side``: direção do sinal original
        - ``barrier_type``: qual barreira foi tocada ('pt', 'sl', 'vertical')
    """
    if pt_sl is None:
        pt_sl = events.attrs.get("pt_sl", labeling_config.pt_sl_ratio)

    pt_mult, sl_mult = pt_sl

    results = []
    close_values = close.values.astype(np.float64)
    close_idx = close.index

    for event_ts, row in events.iterrows():
        t1 = row["t1"]
        trgt = row["trgt"]
        side = row["side"]

        # Localiza índices no array
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

        if start_loc >= len(close_values) or end_loc < start_loc:
            continue

        # Preço de entrada
        entry_price = close_values[start_loc]

        # Calcula barreiras absolutas
        upper = trgt * pt_mult if pt_mult > 0 else np.inf
        lower = -trgt * sl_mult if sl_mult > 0 else -np.inf

        # Encontra primeira barreira tocada
        touch_ts, touch_ret, barrier_type = _find_first_touch(
            close_values=close_values,
            start=start_loc,
            end=end_loc,
            entry_price=entry_price,
            side=int(side),
            upper=upper,
            lower=lower,
        )

        # Converte índice de volta para timestamp
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

    # Estatísticas
    if len(result_df) > 0:
        counts = result_df["barrier_type"].value_counts()
        logger.info(
            "Tripla Barreira: {} eventos | pt={}, sl={}, vertical={}",
            len(result_df),
            counts.get("pt", 0),
            counts.get("sl", 0),
            counts.get("vertical", 0),
        )
    else:
        logger.warning("Tripla Barreira: nenhum evento processado")

    return result_df


# ---------------------------------------------------------------------------
# Kernel otimizado
# ---------------------------------------------------------------------------
@njit
def _find_first_touch(
    close_values: np.ndarray,
    start: int,
    end: int,
    entry_price: float,
    side: int,
    upper: float,
    lower: float,
) -> tuple[int, float, int]:
    """
    Encontra a primeira barreira tocada via iteração rápida.

    Returns
    -------
    tuple[int, float, int]
        (index_do_toque, retorno, tipo_barreira)
        tipo_barreira: 0=pt, 1=sl, 2=vertical
    """
    for i in range(start + 1, end + 1):
        # Retorno relativo à entrada, ajustado pela direção
        ret = (close_values[i] / entry_price - 1.0) * side

        # Barreira superior (Take Profit)
        if ret >= upper:
            return i, ret, 0  # pt

        # Barreira inferior (Stop Loss)
        if ret <= lower:
            return i, ret, 1  # sl

    # Barreira vertical (tempo esgotado)
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
    min_return: float | None = None,
) -> pd.DataFrame:
    """
    Pipeline completo: aplica tripla barreira e gera labels.

    Labels:
    - +1: Take Profit atingido (trade lucrativo)
    - -1: Stop Loss atingido (trade perdedor)
    - 0: Barreira vertical (tempo esgotado, retorno insuficiente)

    Parameters
    ----------
    close : pd.Series
        Preços de fechamento.
    events : pd.DataFrame
        Eventos com t1, trgt, side.
    pt_sl : tuple, optional
        Multiplicadores PT/SL.
    min_return : float, optional
        Retorno mínimo para considerar label != 0.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas: ``t1``, ``ret``, ``label``, ``side``,
        ``barrier_type``.
    """
    if min_return is None:
        min_return = labeling_config.min_return

    result = apply_triple_barrier(close, events, pt_sl)

    if len(result) == 0:
        return result

    # Converte barrier_type numérico para string
    result["barrier_type"] = result["barrier_type"].map(_BARRIER_NAMES)

    # Gera labels
    result["label"] = 0
    result.loc[result["barrier_type"] == "pt", "label"] = 1
    result.loc[result["barrier_type"] == "sl", "label"] = -1

    # Barreira vertical: label = sinal do retorno (se > min_return)
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

"""Filtro CUSUM para Amostragem Baseada em Eventos — TradeSystem5000.

Este módulo implementa o filtro de Soma Cumulativa (CUSUM) simétrico, utilizado
para detectar mudanças estruturais significativas na média de uma série
temporal (ex: preços ou retornos).

O filtro CUSUM é uma alternativa robusta à amostragem temporal fixa, permitindo
que o sistema foque o processamento apenas quando há informação relevante
(eventos).

Funcionalidades:
- **cusum_events**: Filtro CUSUM simétrico com threshold fixo.
- **adaptive_cusum_events**: Filtro CUSUM com threshold dinâmico (EWMA Vol).
- Kernels otimizados via Numba para processamento de alta performance.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulo 2.3.6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from numba import njit

from config.settings import feature_config


# ---------------------------------------------------------------------------
# CUSUM com threshold fixo
# ---------------------------------------------------------------------------
def cusum_events(
    close: pd.Series,
    threshold: float | None = None,
) -> pd.DatetimeIndex:
    """Aplica filtro CUSUM simétrico e retorna timestamps de eventos.

    Um evento é registrado quando a mudança cumulativa (positiva ou negativa)
    ultrapassa o ``threshold``. Após cada evento, os acumuladores são resetados.

    Parameters
    ----------
    close : pd.Series
        Série de preços de fechamento com DatetimeIndex.
    threshold : float, optional
        Limiar de mudança para trigger. Default: calculado a partir da config
        (``cusum_threshold_pct / 100 * std_retornos``).

    Returns
    -------
    pd.DatetimeIndex
        Timestamps onde eventos foram detectados.

    """
    if threshold is None:
        # Threshold baseado em percentual da volatilidade dos retornos
        returns_std = close.pct_change().dropna().std()
        threshold = returns_std * feature_config.cusum_threshold_pct
        logger.debug(
            "CUSUM threshold calculado: {:.6f} (pct={}, std_ret={:.6f})",
            threshold,
            feature_config.cusum_threshold_pct,
            returns_std,
        )

    diff = close.diff().dropna()
    values = diff.values.astype(np.float64)

    logger.info("Aplicando filtro CUSUM (threshold={:.6f}, n={})", threshold, len(values))

    event_indices = _cusum_kernel(values, threshold)

    events = diff.index[event_indices]

    logger.success("CUSUM: {} eventos detectados em {} observações", len(events), len(close))
    return events


# ---------------------------------------------------------------------------
# CUSUM com threshold adaptativo (EWMA)
# ---------------------------------------------------------------------------
def adaptive_cusum_events(
    close: pd.Series,
    ewm_span: int | None = None,
    threshold_multiplier: float | None = None,
) -> pd.DatetimeIndex:
    """CUSUM com threshold adaptativo baseado em volatilidade EWMA.

    O threshold varia ao longo do tempo conforme a volatilidade local,
    capturando mais eventos em períodos tranquilos e menos em períodos
    voláteis.

    Parameters
    ----------
    close : pd.Series
        Série de preços de fechamento com DatetimeIndex.
    ewm_span : int, optional
        Span do EWMA para estimar volatilidade. Default: config.
    threshold_multiplier : float, optional
        Multiplicador aplicado à volatilidade EWMA. Default: config.

    Returns
    -------
    pd.DatetimeIndex
        Timestamps de eventos detectados.

    """
    if ewm_span is None:
        ewm_span = feature_config.cusum_ewm_span
    if threshold_multiplier is None:
        threshold_multiplier = feature_config.cusum_threshold_pct

    diff = close.diff().dropna()
    vol = diff.ewm(span=ewm_span, min_periods=max(1, ewm_span // 4)).std()

    # Threshold adaptativo = volatilidade × multiplicador
    adaptive_h = (vol * threshold_multiplier).values.astype(np.float64)
    values = diff.values.astype(np.float64)

    logger.info(
        "Aplicando CUSUM adaptativo (ewm_span={}, mult={}, n={})",
        ewm_span,
        threshold_multiplier,
        len(values),
    )

    event_indices = _adaptive_cusum_kernel(values, adaptive_h)

    events = diff.index[event_indices]

    logger.success(
        "CUSUM adaptativo: {} eventos detectados em {} observações",
        len(events),
        len(close),
    )
    return events


# ---------------------------------------------------------------------------
# Kernels otimizados com Numba
# ---------------------------------------------------------------------------
@njit
def _cusum_kernel(values: np.ndarray, threshold: float) -> list[int]:
    """Kernel CUSUM simétrico com threshold fixo."""
    s_pos = 0.0
    s_neg = 0.0
    events: list[int] = []

    for i in range(len(values)):
        s_pos = max(0.0, s_pos + values[i])
        s_neg = min(0.0, s_neg + values[i])

        if s_pos > threshold:
            events.append(i)
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg < -threshold:
            events.append(i)
            s_pos = 0.0
            s_neg = 0.0

    return events


@njit
def _adaptive_cusum_kernel(values: np.ndarray, thresholds: np.ndarray) -> list[int]:
    """Kernel CUSUM simétrico com threshold variável no tempo."""
    s_pos = 0.0
    s_neg = 0.0
    events: list[int] = []

    for i in range(len(values)):
        h = thresholds[i]

        # Ignora onde threshold é NaN ou zero
        if h != h or h <= 0.0:  # NaN check via h != h
            continue

        s_pos = max(0.0, s_pos + values[i])
        s_neg = min(0.0, s_neg + values[i])

        if s_pos > h:
            events.append(i)
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg < -h:
            events.append(i)
            s_pos = 0.0
            s_neg = 0.0

    return events

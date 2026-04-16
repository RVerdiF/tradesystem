"""Volatilidade Dinâmica para Labeling — TradeSystem5000.

Este módulo calcula a volatilidade dos retornos utilizando EWMA (Exponentially
Weighted Moving Average) para ajuste dinâmico das barreiras de saída.

O ajuste dinâmico garante que os alvos de lucro e limites de perda sejam
proporcionais ao regime de risco atual do mercado, mantendo a consistência
estatística dos rótulos.

Funcionalidades:
- **daily_vol**: Estimativa de volatilidade via EWMA dos retornos.
- **get_volatility_targets**: Mapeamento da volatilidade para timestamps de eventos.
- **vol_regime**: Identificação de regimes de aceleração de volatilidade.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulo 3.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import labeling_config


# ---------------------------------------------------------------------------
# Volatilidade EWMA dos retornos
# ---------------------------------------------------------------------------
def daily_vol(
    close: pd.Series,
    span: int | None = None,
    log_returns: bool = True,
) -> pd.Series:
    """Estima a volatilidade dinâmica via desvio padrão EWMA dos retornos.

    Usado para definir a largura das barreiras no método da Tripla Barreira.
    Cada barra recebe um alvo de volatilidade proporcional à volatilidade
    recente do ativo.

    Parameters
    ----------
    close : pd.Series
        Série de preços de fechamento com DatetimeIndex.
    span : int, optional
        Span do EWMA. Default: config (``vol_span``).
    log_returns : bool
        Se True, usa log-retornos. Se False, retornos simples.

    Returns
    -------
    pd.Series
        Volatilidade estimada (sempre positiva), com mesmo índice.

    Notes
    -----
    O uso de EWMA dá mais peso a observações recentes, capturando mudanças
    de regime de volatilidade mais rapidamente que uma janela rolling fixa.

    """
    if span is None:
        span = labeling_config.vol_span

    if log_returns:
        returns = np.log(close / close.shift(1))
    else:
        returns = close.pct_change()

    # EWMA do desvio padrão
    vol = returns.ewm(span=span, min_periods=max(1, span // 4)).std()
    vol.name = "volatility"

    logger.debug(
        "Volatilidade EWMA (span={}): média={:.6f}, atual={:.6f}",
        span,
        vol.mean(),
        vol.iloc[-1] if len(vol) > 0 else 0.0,
    )
    return vol


# ---------------------------------------------------------------------------
# Targets de volatilidade para eventos
# ---------------------------------------------------------------------------
def get_volatility_targets(
    close: pd.Series,
    event_timestamps: pd.DatetimeIndex,
    span: int | None = None,
) -> pd.Series:
    """Retorna a volatilidade estimada nos instantes de cada evento.

    Parameters
    ----------
    close : pd.Series
        Série de preços.
    event_timestamps : pd.DatetimeIndex
        Timestamps dos eventos (gerados pelo CUSUM ou Alpha).
    span : int, optional
        Span da volatilidade EWMA.

    Returns
    -------
    pd.Series
        Volatilidade nos timestamps dos eventos. Índice = event_timestamps.

    """
    vol = daily_vol(close, span=span)

    # Reindexar para os timestamps dos eventos com forward fill
    targets = vol.reindex(event_timestamps, method="ffill")
    targets.name = "target"

    # Remove eventos sem volatilidade estimada (início da série)
    n_nan = targets.isna().sum()
    if n_nan > 0:
        logger.warning(
            "Volatilidade: {} eventos sem estimativa (início da série). Removidos.",
            n_nan,
        )
        targets = targets.dropna()

    logger.info(
        "Targets de volatilidade: {} eventos, vol média={:.6f}",
        len(targets),
        targets.mean() if len(targets) > 0 else 0.0,
    )
    return targets

"""
Amostragem de Barras Alternativas — TradeSystem5000.

Este módulo transforma dados brutos de ticks em barras OHLCV baseadas em
critérios alternativos ao tempo (Volume e Valor Financeiro/Dólar).

A amostragem alternativa visa reduzir a heterocedasticidade e normalizar a
distribuição dos retornos, propriedades fundamentais para o sucesso de modelos
de Machine Learning em finanças.

Funcionalidades:
- **volume_bars**: Barras geradas por volume acumulado.
- **dollar_bars**: Barras geradas por valor financeiro acumulado (Preço x Volume).
- **tick_bars**: Barras geradas por contagem fixa de ticks.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulo 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from numba import njit

from config.settings import bar_sampling_config


# ---------------------------------------------------------------------------
# Barras de Volume
# ---------------------------------------------------------------------------
def volume_bars(
    ticks: pd.DataFrame,
    threshold: int | None = None,
    price_col: str = "last",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Gera barras OHLCV amostradas por volume acumulado.

    Uma nova barra é criada quando o volume acumulado desde a última barra
    atinge o ``threshold``.

    Parameters
    ----------
    ticks : pd.DataFrame
        DataFrame de ticks com colunas de preço e volume.
    threshold : int, optional
        Volume acumulado para gerar nova barra. Default: config.
    price_col : str
        Coluna de preço.
    volume_col : str
        Coluna de volume.

    Returns
    -------
    pd.DataFrame
        Barras OHLCV com DatetimeIndex (timestamp de abertura da barra).
    """
    if threshold is None:
        threshold = bar_sampling_config.volume_bar_threshold

    if price_col not in ticks.columns or volume_col not in ticks.columns:
        raise ValueError(
            f"Colunas necessárias: '{price_col}' e '{volume_col}'. "
            f"Colunas disponíveis: {list(ticks.columns)}"
        )

    logger.info("Gerando barras de volume (threshold={})", threshold)

    prices = ticks[price_col].values.astype(np.float64)
    volumes = ticks[volume_col].values.astype(np.float64)
    timestamps = ticks.index.values

    # Usa Numba para performance
    indices = _sample_by_cumulative(volumes, float(threshold))

    bars = _build_bars(prices, volumes, timestamps, indices)

    logger.success("Barras de volume: {} barras geradas a partir de {} ticks", len(bars), len(ticks))
    return bars


# ---------------------------------------------------------------------------
# Barras de Dólar (Dollar Bars)
# ---------------------------------------------------------------------------
def dollar_bars(
    ticks: pd.DataFrame,
    threshold: float | None = None,
    price_col: str = "last",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Gera barras OHLCV amostradas por valor financeiro acumulado (preço × volume).

    Uma nova barra é criada quando o valor financeiro acumulado atinge o ``threshold``.

    Parameters
    ----------
    ticks : pd.DataFrame
        DataFrame de ticks com colunas de preço e volume.
    threshold : float, optional
        Valor financeiro acumulado para gerar nova barra. Default: config.
    price_col : str
        Coluna de preço.
    volume_col : str
        Coluna de volume.

    Returns
    -------
    pd.DataFrame
        Barras OHLCV com DatetimeIndex.
    """
    if threshold is None:
        threshold = bar_sampling_config.dollar_bar_threshold

    if price_col not in ticks.columns or volume_col not in ticks.columns:
        raise ValueError(
            f"Colunas necessárias: '{price_col}' e '{volume_col}'. "
            f"Colunas disponíveis: {list(ticks.columns)}"
        )

    logger.info("Gerando barras de dólar (threshold={:,.0f})", threshold)

    prices = ticks[price_col].values.astype(np.float64)
    volumes = ticks[volume_col].values.astype(np.float64)
    timestamps = ticks.index.values

    # Valor financeiro = preço × volume
    dollar_values = prices * volumes

    indices = _sample_by_cumulative(dollar_values, threshold)

    bars = _build_bars(prices, volumes, timestamps, indices)

    logger.success(
        "Barras de dólar: {} barras geradas a partir de {} ticks", len(bars), len(ticks)
    )
    return bars


# ---------------------------------------------------------------------------
# Barras de Tick (Tick Bars)
# ---------------------------------------------------------------------------
def tick_bars(
    ticks: pd.DataFrame,
    threshold: int = 100,
    price_col: str = "last",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Gera barras OHLCV amostradas por contagem de ticks.

    Uma nova barra é criada a cada ``threshold`` ticks.

    Parameters
    ----------
    ticks : pd.DataFrame
        DataFrame de ticks.
    threshold : int
        Número de ticks por barra.
    price_col : str
        Coluna de preço.
    volume_col : str
        Coluna de volume.

    Returns
    -------
    pd.DataFrame
        Barras OHLCV com DatetimeIndex.
    """
    logger.info("Gerando barras de tick (threshold={})", threshold)

    prices = ticks[price_col].values.astype(np.float64)
    volumes = ticks[volume_col].values.astype(np.float64)
    timestamps = ticks.index.values

    # Cada tick contribui com 1.0
    ones = np.ones(len(prices), dtype=np.float64)
    indices = _sample_by_cumulative(ones, float(threshold))

    bars = _build_bars(prices, volumes, timestamps, indices)

    logger.success(
        "Barras de tick: {} barras geradas a partir de {} ticks", len(bars), len(ticks)
    )
    return bars


# ---------------------------------------------------------------------------
# Funções internas otimizadas com Numba
# ---------------------------------------------------------------------------
@njit
def _sample_by_cumulative(values: np.ndarray, threshold: float) -> list[int]:
    """
    Encontra os índices onde a soma cumulativa atinge o threshold.

    Retorna a lista de índices de final de barra.
    """
    indices: list[int] = []
    cum_sum = 0.0

    for i in range(len(values)):
        cum_sum += values[i]
        if cum_sum >= threshold:
            indices.append(i)
            cum_sum = 0.0

    return indices


def _build_bars(
    prices: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,
    bar_end_indices: list[int],
) -> pd.DataFrame:
    """
    Constrói DataFrame OHLCV a partir dos índices de fim de barra.

    Parameters
    ----------
    prices : np.ndarray
        Array de preços.
    volumes : np.ndarray
        Array de volumes.
    timestamps : np.ndarray
        Array de timestamps.
    bar_end_indices : list[int]
        Índices onde cada barra termina.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas: open, high, low, close, volume, n_ticks.
    """
    if len(bar_end_indices) == 0:
        logger.warning("Nenhuma barra gerada. Threshold muito alto?")
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "n_ticks"]
        )

    bars: list[dict] = []
    start_idx = 0

    for end_idx in bar_end_indices:
        segment_prices = prices[start_idx : end_idx + 1]
        segment_volumes = volumes[start_idx : end_idx + 1]

        bar = {
            "time": timestamps[start_idx],  # timestamp de abertura
            "open": segment_prices[0],
            "high": segment_prices.max(),
            "low": segment_prices.min(),
            "close": segment_prices[-1],
            "volume": segment_volumes.sum(),
            "n_ticks": len(segment_prices),
        }
        bars.append(bar)
        start_idx = end_idx + 1

    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    return df

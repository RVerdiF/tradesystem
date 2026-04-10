"""
Extração de Dados do MetaTrader 5 — TradeSystem5000.

Este módulo gerencia o download automatizado de dados históricos (ticks e barras)
diretamente do terminal MetaTrader 5, com suporte a download incremental.

Funcionalidades:
- **extract_ticks**: Download de histórico de ticks em range temporal.
- **extract_ohlc**: Download de barras OHLCV por posição ou data.
- **extract_ohlc_incremental**: Sincronização eficiente de dados locais com o servidor.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from src.data.mt5_connector import MT5Connector

from config.settings import DEFAULT_TIMEFRAME

# Mapeamento de timeframes para constantes MT5
TIMEFRAME_MAP: dict[int, str] = {
    1: "M1",
    5: "M5",
    15: "M15",
    30: "M30",
    60: "H1",
    240: "H4",
    1440: "D1",
    10080: "W1",
    43200: "MN1",
}

# Mapeamento reverso de strings CLI (yfinance style) para timeframes MT5
INTERVAL_TO_TF: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "60m": 60,
    "4h": 240,
    "1d": 1440,
    "1wk": 10080,
    "1mo": 43200,
}


class DataExtractionError(Exception):
    """Erro na extração de dados do MT5."""


# ---------------------------------------------------------------------------
# Extração de Ticks
# ---------------------------------------------------------------------------
def extract_ticks(
    symbol: str,
    date_from: datetime,
    date_to: datetime,
    flags: int | None = None,
) -> pd.DataFrame:
    """
    Baixa ticks históricos via ``mt5.copy_ticks_range``.

    Parameters
    ----------
    symbol : str
        Símbolo do ativo (ex: "PETR4", "WINFUT").
    date_from : datetime
        Data/hora inicial (UTC).
    date_to : datetime
        Data/hora final (UTC).
    flags : int, optional
        Flags MT5 (COPY_TICKS_ALL, COPY_TICKS_INFO, COPY_TICKS_TRADE).
        Default: COPY_TICKS_ALL.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas: time, bid, ask, last, volume, flags.
    """
    if not MT5_AVAILABLE:
        raise DataExtractionError("MetaTrader5 não disponível.")

    if flags is None:
        flags = mt5.COPY_TICKS_ALL

    # Garante timezone UTC
    if date_from.tzinfo is None:
        date_from = date_from.replace(tzinfo=timezone.utc)
    if date_to.tzinfo is None:
        date_to = date_to.replace(tzinfo=timezone.utc)

    logger.info(
        "Extraindo ticks de {} | {} → {}",
        symbol,
        date_from.isoformat(),
        date_to.isoformat(),
    )

    ticks = mt5.copy_ticks_range(symbol, date_from, date_to, flags)

    if ticks is None or len(ticks) == 0:
        error = mt5.last_error()
        raise DataExtractionError(
            f"Nenhum tick retornado para {symbol}. Erro MT5: {error}"
        )

    df = pd.DataFrame(ticks)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    logger.success("Ticks de {}: {} registros extraídos", symbol, len(df))
    return df


# ---------------------------------------------------------------------------
# Extração de Barras OHLC
# ---------------------------------------------------------------------------
def extract_ohlc(
    symbol: str,
    timeframe: int = DEFAULT_TIMEFRAME,
    n_bars: int = 1000,
    date_from: datetime | None = None,
) -> pd.DataFrame:
    """
    Baixa barras OHLCV históricas via ``mt5.copy_rates_from_pos`` ou
    ``mt5.copy_rates_from``.

    Parameters
    ----------
    symbol : str
        Símbolo do ativo.
    timeframe : int
        Timeframe MT5 (1=M1, 5=M5, 15=M15, 60=H1, 1440=D1).
    n_bars : int
        Quantidade de barras a retornar (usado com copy_rates_from_pos).
    date_from : datetime, optional
        Se fornecido, usa ``copy_rates_from`` a partir desta data.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas: time, open, high, low, close, tick_volume,
        spread, real_volume.
    """
    if not MT5_AVAILABLE:
        raise DataExtractionError("MetaTrader5 não disponível.")

    tf_name = TIMEFRAME_MAP.get(timeframe, f"TF{timeframe}")
    logger.info("Extraindo OHLC de {} | {} | {} barras", symbol, tf_name, n_bars)

    # Mapeia int → constante MT5
    mt5_tf = _resolve_timeframe(timeframe)

    if date_from is not None:
        if date_from.tzinfo is None:
            date_from = date_from.replace(tzinfo=timezone.utc)
        rates = mt5.copy_rates_from(symbol, mt5_tf, date_from, n_bars)
    else:
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, n_bars)

    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        raise DataExtractionError(
            f"Nenhuma barra retornada para {symbol} ({tf_name}). Erro MT5: {error}"
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    # Validação básica de integridade
    _validate_ohlc(df, symbol)

    logger.success("OHLC de {} ({}): {} barras extraídas", symbol, tf_name, len(df))
    return df


# ---------------------------------------------------------------------------
# Download Incremental
# ---------------------------------------------------------------------------
def extract_ohlc_incremental(
    symbol: str,
    existing_df: pd.DataFrame | None,
    timeframe: int = DEFAULT_TIMEFRAME,
    n_bars: int = 1000,
) -> pd.DataFrame:
    """
    Baixa apenas barras novas (posteriores ao último registro existente).

    Parameters
    ----------
    symbol : str
        Símbolo do ativo.
    existing_df : pd.DataFrame or None
        DataFrame com dados já existentes. Se None, baixa tudo.
    timeframe : int
        Timeframe MT5.
    n_bars : int
        Quantidade máxima de barras novas.

    Returns
    -------
    pd.DataFrame
        DataFrame combinado (existentes + novos), sem duplicatas.
    """
    if existing_df is not None and len(existing_df) > 0:
        last_time = existing_df.index.max()
        logger.info("Download incremental de {} a partir de {}", symbol, last_time)
        new_df = extract_ohlc(symbol, timeframe, n_bars, date_from=last_time)
        # Combina e remove duplicatas
        combined = pd.concat([existing_df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        return combined
    else:
        return extract_ohlc(symbol, timeframe, n_bars)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_timeframe(tf: int) -> int:
    """Converte inteiro para constante MT5 de timeframe."""
    if not MT5_AVAILABLE:
        return tf

    tf_map = {
        1: mt5.TIMEFRAME_M1,
        5: mt5.TIMEFRAME_M5,
        15: mt5.TIMEFRAME_M15,
        30: mt5.TIMEFRAME_M30,
        60: mt5.TIMEFRAME_H1,
        240: mt5.TIMEFRAME_H4,
        1440: mt5.TIMEFRAME_D1,
        10080: mt5.TIMEFRAME_W1,
        43200: mt5.TIMEFRAME_MN1,
    }
    return tf_map.get(tf, tf)


def _validate_ohlc(df: pd.DataFrame, symbol: str) -> None:
    """Validação básica de integridade OHLC."""
    issues = []

    # High >= Low
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl > 0:
        issues.append(f"{bad_hl} barras com high < low")

    # Open e Close dentro de [Low, High]
    bad_open = ((df["open"] < df["low"]) | (df["open"] > df["high"])).sum()
    if bad_open > 0:
        issues.append(f"{bad_open} barras com open fora de [low, high]")

    bad_close = ((df["close"] < df["low"]) | (df["close"] > df["high"])).sum()
    if bad_close > 0:
        issues.append(f"{bad_close} barras com close fora de [low, high]")

    # NaN check
    nan_count = df[["open", "high", "low", "close"]].isna().sum().sum()
    if nan_count > 0:
        issues.append(f"{nan_count} valores NaN encontrados")

    if issues:
        for issue in issues:
            logger.warning("Validação OHLC [{}]: {}", symbol, issue)
    else:
        logger.debug("Validação OHLC [{}]: OK", symbol)

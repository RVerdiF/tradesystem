"""
Limpeza e Saneamento de Dados — TradeSystem5000.

Este módulo fornece ferramentas para filtrar ruídos, remover spikes (bad ticks)
e validar a integridade estrutural de dados OHLCV e ticks.

Funcionalidades:
- **remove_spikes**: Filtro baseado em Z-score rolling para preços.
- **remove_tick_spikes**: Filtro de retornos impossíveis entre ticks.
- **fill_gaps**: Preenchimento de lacunas temporais (ffill, interpolation).
- **validate_ohlc**: Verificação de consistência lógica (High >= Low, etc).

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulo 3 (Data Cleaning).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import cleaning_config


# ---------------------------------------------------------------------------
# Remoção de Spikes (Bad Ticks)
# ---------------------------------------------------------------------------
def remove_spikes(
    df: pd.DataFrame,
    price_col: str = "close",
    z_threshold: float | None = None,
    window: int = 50,
) -> pd.DataFrame:
    """
    Remove registros cujo preço é um outlier estatístico (spike).

    Usa Z-score rolling para identificar valores que desviam significativamente
    da média móvel local.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com coluna de preço.
    price_col : str
        Nome da coluna de preço a analisar.
    z_threshold : float, optional
        Limiar de Z-score para considerar spike. Default: config.
    window : int
        Janela da média/desvio móvel.

    Returns
    -------
    pd.DataFrame
        DataFrame sem os spikes.
    """
    if z_threshold is None:
        z_threshold = cleaning_config.spike_z_threshold

    original_len = len(df)

    if price_col not in df.columns:
        logger.warning("Coluna '{}' não encontrada. Pulando remoção de spikes.", price_col)
        return df

    prices = df[price_col]
    rolling_mean = prices.rolling(window=window, min_periods=max(1, window // 4)).mean()
    rolling_std = prices.rolling(window=window, min_periods=max(1, window // 4)).std()

    # Evita divisão por zero
    rolling_std = rolling_std.replace(0, np.nan)

    z_scores = ((prices - rolling_mean) / rolling_std).abs()

    # Mantém os primeiros registros que não tem janela suficiente
    mask = z_scores.isna() | (z_scores <= z_threshold)
    cleaned = df[mask].copy()

    removed = original_len - len(cleaned)
    if removed > 0:
        logger.info(
            "Spikes removidos: {} de {} registros ({:.2f}%)",
            removed,
            original_len,
            100 * removed / original_len,
        )
    else:
        logger.debug("Nenhum spike detectado (threshold Z={})", z_threshold)

    return cleaned


# ---------------------------------------------------------------------------
# Remoção de Spikes por Retornos (para ticks)
# ---------------------------------------------------------------------------
def remove_tick_spikes(
    df: pd.DataFrame,
    price_col: str = "last",
    max_return_pct: float = 1.0,
) -> pd.DataFrame:
    """
    Remove ticks com retornos absolutos impossíveis (ex: >1% de um tick para outro).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de ticks.
    price_col : str
        Coluna de preço.
    max_return_pct : float
        Retorno máximo aceitável entre ticks consecutivos (em %).

    Returns
    -------
    pd.DataFrame
        DataFrame sem ticks espúrios.
    """
    original_len = len(df)

    if price_col not in df.columns:
        return df

    returns = df[price_col].pct_change().abs()
    threshold = max_return_pct / 100.0

    mask = returns.isna() | (returns <= threshold)
    cleaned = df[mask].copy()

    removed = original_len - len(cleaned)
    if removed > 0:
        logger.info(
            "Tick spikes removidos: {} (retorno > {:.2f}%)",
            removed,
            max_return_pct,
        )

    return cleaned


# ---------------------------------------------------------------------------
# Preenchimento de Lacunas
# ---------------------------------------------------------------------------
def fill_gaps(
    df: pd.DataFrame,
    method: str | None = None,
    max_gap: int | None = None,
    freq: str | None = None,
) -> pd.DataFrame:
    """
    Preenche lacunas temporais no DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com DatetimeIndex.
    method : str, optional
        Método de preenchimento: ``ffill`` (forward fill), ``bfill``,
        ``interpolate``. Default: config.
    max_gap : int, optional
        Quantidade máxima de períodos a preencher. Lacunas maiores são
        mantidas como NaN. Default: config.
    freq : str, optional
        Frequência para reindexação (ex: "5min", "1min"). Se None,
        infere a frequência do DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame com lacunas preenchidas.
    """
    if method is None:
        method = cleaning_config.fill_method
    if max_gap is None:
        max_gap = cleaning_config.max_gap_seconds

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Index não é DatetimeIndex. Pulando preenchimento de lacunas.")
        return df

    # Detecta frequência se não fornecida
    if freq is None:
        freq = pd.infer_freq(df.index)
        if freq is None:
            logger.debug("Frequência não detectável. Pulando reindexação.")
            return _fill_existing_nans(df, method, max_gap)

    # Reindexação para frequência regular
    original_len = len(df)
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_index)

    gaps_created = len(df) - original_len
    if gaps_created > 0:
        logger.info("Lacunas detectadas: {} períodos faltantes", gaps_created)

    return _fill_existing_nans(df, method, max_gap)


def _fill_existing_nans(df: pd.DataFrame, method: str, max_gap: int) -> pd.DataFrame:
    """Preenche NaNs existentes conforme o método especificado."""
    nan_before = df.isna().sum().sum()

    if method == "ffill":
        df = df.ffill(limit=max_gap)
    elif method == "bfill":
        df = df.bfill(limit=max_gap)
    elif method == "interpolate":
        df = df.interpolate(method="time", limit=max_gap)
    else:
        logger.warning("Método de preenchimento desconhecido: '{}'. Usando ffill.", method)
        df = df.ffill(limit=max_gap)

    nan_after = df.isna().sum().sum()
    filled = nan_before - nan_after
    if filled > 0:
        logger.info("NaNs preenchidos: {} (restantes: {})", filled, nan_after)

    return df


# ---------------------------------------------------------------------------
# Validação OHLC
# ---------------------------------------------------------------------------
def validate_ohlc(df: pd.DataFrame, fix: bool = False) -> pd.DataFrame:
    """
    Valida e opcionalmente corrige dados OHLC.

    Verificações:
    - High >= Low
    - Open e Close dentro de [Low, High]
    - Sem valores negativos
    - Sem volumes negativos

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas open, high, low, close.
    fix : bool
        Se True, corrige inconsistências automaticamente.

    Returns
    -------
    pd.DataFrame
        DataFrame validado (e possivelmente corrigido).
    """
    df = df.copy()
    issues: list[str] = []

    # --- High >= Low ---
    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        count = bad_hl.sum()
        issues.append(f"{count} barras com high < low")
        if fix:
            # Swap high/low
            df.loc[bad_hl, ["high", "low"]] = df.loc[bad_hl, ["low", "high"]].values

    # --- Open em [Low, High] ---
    bad_open = (df["open"] < df["low"]) | (df["open"] > df["high"])
    if bad_open.any():
        count = bad_open.sum()
        issues.append(f"{count} barras com open fora de [low, high]")
        if fix:
            df.loc[bad_open, "open"] = df.loc[bad_open, "open"].clip(
                lower=df.loc[bad_open, "low"],
                upper=df.loc[bad_open, "high"],
            )

    # --- Close em [Low, High] ---
    bad_close = (df["close"] < df["low"]) | (df["close"] > df["high"])
    if bad_close.any():
        count = bad_close.sum()
        issues.append(f"{count} barras com close fora de [low, high]")
        if fix:
            df.loc[bad_close, "close"] = df.loc[bad_close, "close"].clip(
                lower=df.loc[bad_close, "low"],
                upper=df.loc[bad_close, "high"],
            )

    # --- Valores negativos ---
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                issues.append(f"{neg} valores negativos em {col}")
                if fix:
                    df.loc[df[col] < 0, col] = np.nan

    # --- Volume negativo ---
    vol_cols = [c for c in df.columns if "volume" in c.lower()]
    for col in vol_cols:
        neg = (df[col] < 0).sum()
        if neg > 0:
            issues.append(f"{neg} volumes negativos em {col}")
            if fix:
                df.loc[df[col] < 0, col] = 0

    # --- Resumo ---
    if issues:
        action = "Corrigidos" if fix else "Encontrados"
        for issue in issues:
            logger.warning("Validação OHLC — {}: {}", action, issue)
    else:
        logger.debug("Validação OHLC: todos os dados OK")

    return df


# ---------------------------------------------------------------------------
# Pipeline completo de limpeza
# ---------------------------------------------------------------------------
def clean_ohlc(
    df: pd.DataFrame,
    z_threshold: float | None = None,
    fix_ohlc: bool = True,
    fill_method: str | None = None,
    max_gap: int | None = None,
) -> pd.DataFrame:
    """
    Pipeline completo de limpeza para dados OHLC.

    Executa na ordem:
    1. Validação/correção OHLC
    2. Remoção de spikes
    3. Preenchimento de lacunas

    Returns
    -------
    pd.DataFrame
        DataFrame limpo.
    """
    logger.info("Iniciando pipeline de limpeza ({} registros)", len(df))

    # 1. Validação
    df = validate_ohlc(df, fix=fix_ohlc)

    # 2. Remoção de spikes
    df = remove_spikes(df, price_col="close", z_threshold=z_threshold)

    # 3. Preenchimento de lacunas
    df = fill_gaps(df, method=fill_method, max_gap=max_gap)

    logger.success("Limpeza concluída: {} registros finais", len(df))
    return df


def clean_ticks(
    df: pd.DataFrame,
    max_return_pct: float = 1.0,
) -> pd.DataFrame:
    """
    Pipeline de limpeza para dados de ticks.

    Executa:
    1. Remoção de ticks com retornos impossíveis
    2. Remoção de duplicatas

    Returns
    -------
    pd.DataFrame
        DataFrame de ticks limpo.
    """
    logger.info("Limpando ticks ({} registros)", len(df))

    # Remove duplicatas
    original_len = len(df)
    df = df[~df.index.duplicated(keep="last")]
    dupes = original_len - len(df)
    if dupes > 0:
        logger.info("Duplicatas removidas: {}", dupes)

    # Remove spikes
    df = remove_tick_spikes(df, max_return_pct=max_return_pct)

    logger.success("Ticks limpos: {} registros finais", len(df))
    return df

"""
2.4 — Normalização Temporal.

Normaliza features com métodos point-in-time para eliminar diferenças de
escala entre indicadores, sem introduzir look-ahead bias.

Referência: López de Prado, *Advances in Financial Machine Learning*, Cap. 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import feature_config


# ---------------------------------------------------------------------------
# Z-Score Móvel
# ---------------------------------------------------------------------------
def rolling_zscore(
    series: pd.Series,
    window: int | None = None,
) -> pd.Series:
    """
    Z-score com janela móvel (point-in-time).

    Parameters
    ----------
    series : pd.Series
        Série a normalizar.
    window : int, optional
        Tamanho da janela. Default: config.

    Returns
    -------
    pd.Series
        Série normalizada (média ≈ 0, std ≈ 1 na janela).
    """
    if window is None:
        window = feature_config.zscore_window

    min_periods = max(1, window // 4)
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()

    # Evita divisão por zero
    rolling_std = rolling_std.replace(0, np.nan)

    result = (series - rolling_mean) / rolling_std
    result.name = f"{series.name}_zscore" if series.name else "zscore"
    return result


# ---------------------------------------------------------------------------
# Rank Percentual Expandido
# ---------------------------------------------------------------------------
def expanding_rank(series: pd.Series) -> pd.Series:
    """
    Rank percentual expandido (de 0 a 1).

    Para cada timestamp ``t``, retorna a posição relativa do valor atual
    em relação a todos os valores anteriores (inclusive).

    Parameters
    ----------
    series : pd.Series
        Série a ranquear.

    Returns
    -------
    pd.Series
        Série com valores em [0, 1].
    """
    result = series.expanding().rank(pct=True)
    result.name = f"{series.name}_rank" if series.name else "rank"
    return result


# ---------------------------------------------------------------------------
# Normalização em lote
# ---------------------------------------------------------------------------
def normalize_features(
    df: pd.DataFrame,
    method: str = "zscore",
    window: int | None = None,
) -> pd.DataFrame:
    """
    Normaliza todas as colunas numéricas de um DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com features.
    method : str
        Método: ``zscore`` (rolling Z-score) ou ``rank`` (expanding rank).
    window : int, optional
        Janela para Z-score. Default: config.

    Returns
    -------
    pd.DataFrame
        DataFrame normalizado com mesmas colunas e índice.
    """
    if window is None:
        window = feature_config.zscore_window

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logger.info("Normalizando {} colunas (método={})", len(numeric_cols), method)

    result = df.copy()

    for col in numeric_cols:
        if method == "zscore":
            result[col] = rolling_zscore(df[col], window=window)
        elif method == "rank":
            result[col] = expanding_rank(df[col])
        else:
            logger.warning("Método desconhecido: '{}'. Usando zscore.", method)
            result[col] = rolling_zscore(df[col], window=window)

    logger.success("Normalização concluída ({} colunas)", len(numeric_cols))
    return result


# ---------------------------------------------------------------------------
# Validação anti look-ahead
# ---------------------------------------------------------------------------
def validate_no_lookahead(
    normalized: pd.DataFrame,
    original: pd.DataFrame,
    window: int | None = None,
) -> bool:
    """
    Verifica que a normalização não introduziu look-ahead bias.

    Para cada timestamp ``t``, recalcula o Z-score usando apenas dados ``≤ t``
    e compara com o valor no DataFrame normalizado.

    Parameters
    ----------
    normalized : pd.DataFrame
        DataFrame normalizado a validar.
    original : pd.DataFrame
        DataFrame original (pré-normalização).
    window : int, optional
        Janela usada na normalização.

    Returns
    -------
    bool
        True se não há look-ahead bias, False caso contrário.
    """
    if window is None:
        window = feature_config.zscore_window

    numeric_cols = original.select_dtypes(include=[np.number]).columns

    # Testa um subconjunto de pontos para performance
    n_tests = min(50, len(original))
    test_indices = np.linspace(window, len(original) - 1, n_tests, dtype=int)

    for col in numeric_cols:
        if col not in normalized.columns:
            continue

        for idx in test_indices:
            # Recalcula usando apenas dados até idx (inclusive)
            subset = original[col].iloc[: idx + 1]
            recent = subset.iloc[-window:] if len(subset) >= window else subset

            mean_val = recent.mean()
            std_val = recent.std()

            if std_val == 0 or np.isnan(std_val):
                continue

            expected = (original[col].iloc[idx] - mean_val) / std_val
            actual = normalized[col].iloc[idx]

            if np.isnan(expected) or np.isnan(actual):
                continue

            if not np.isclose(expected, actual, atol=1e-10):
                logger.error(
                    "Look-ahead detectado em col='{}', idx={}: esperado={:.6f}, obtido={:.6f}",
                    col,
                    idx,
                    expected,
                    actual,
                )
                return False

    logger.debug("Validação anti look-ahead: OK")
    return True

"""Normalização Temporal de Features — TradeSystem5000.

Este módulo implementa técnicas de normalização point-in-time para garantir
que as features tenham escalas comparáveis sem introduzir viés de antecipação
(look-ahead bias).

Funcionalidades:
- **rolling_zscore**: Normalização gaussiana via média e desvio móvel.
- **expanding_rank**: Rank percentual expandido (estabilidade em caudas longas).
- **normalize_features**: Processamento em lote de múltiplos métodos.
- **validate_no_lookahead**: Verificação rigorosa contra vazamento de dados futuros.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulo 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import feature_config

# Colunas produzidas pelo normalizer (rolling z-score / expanding rank).
# Usado por validate_no_lookahead para escopo de validação anti-lookahead.
# Nomes devem bater com as features de compute_all_features + sufixo _zscore/_rank.
ROLLING_NORMALIZED_COLS = [
    "ma_dist_fast_zscore",
    "ma_dist_slow_zscore",
    "roc_zscore",
    "atr_zscore",
    "rolling_vol_zscore",
    "garman_klass_zscore",
    "skew_zscore",
    "kurt_zscore",
    "ofi_zscore",
    "vpin_zscore",
    "vsa_rel_spread_zscore",
    "vsa_bar_pos_zscore",
    "vsa_rel_vol_zscore",
    "vsa_wick_ratio_zscore",
]

# Colunas brutas que NUNCA devem ser normalizadas.
# Inclui OHLCV raw e campos auxiliares de tempo.
_BLOCKLIST_RAW_COLS = frozenset(
    [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "time",
        "tick_volume",
        "real_volume",
    ]
)


# ---------------------------------------------------------------------------
# Z-Score Móvel
# ---------------------------------------------------------------------------
def rolling_zscore(
    series: pd.Series,
    window: int | None = None,
) -> pd.Series:
    """Z-score com janela móvel (point-in-time).

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
    """Rank percentual expandido (de 0 a 1).

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
    """Normaliza colunas de features de um DataFrame, excluindo colunas brutas OHLCV.

    Colunas em ``_BLOCKLIST_RAW_COLS`` (open, high, low, close, volume e
    auxiliares de tempo) são preservadas sem transformação. Todas as demais
    colunas numéricas recebem a normalização point-in-time especificada.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com features. Colunas OHLCV devem estar em lowercase
        (padrão da API MT5 Python).
    method : str
        Método: ``zscore`` (rolling Z-score) ou ``rank`` (expanding rank).
    window : int, optional
        Janela para Z-score. Default: config.

    Returns
    -------
    pd.DataFrame
        DataFrame com features normalizadas e colunas OHLCV inalteradas.

    """
    if window is None:
        window = feature_config.zscore_window

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclui colunas brutas OHLCV e auxiliares — normalizar preço gera
    # ruído e features inúteis para o modelo (data leakage / redundância).
    target_cols = [c for c in numeric_cols if c not in _BLOCKLIST_RAW_COLS]
    skipped = set(numeric_cols) - set(target_cols)
    if skipped:
        logger.info("Normalização: colunas ignoradas (raw OHLCV): {}", sorted(skipped))
    logger.info("Normalizando {} colunas (método={})", len(target_cols), method)

    result = df.copy()

    for col in target_cols:
        if method == "zscore":
            result[col] = rolling_zscore(df[col], window=window)
        elif method == "rank":
            result[col] = expanding_rank(df[col])
        else:
            logger.warning("Método desconhecido: '{}'. Usando zscore.", method)
            result[col] = rolling_zscore(df[col], window=window)

    logger.success("Normalização concluída ({} colunas)", len(target_cols))
    return result


# ---------------------------------------------------------------------------
# Validação anti look-ahead
# ---------------------------------------------------------------------------
def validate_no_lookahead(
    normalized: pd.DataFrame,
    original: pd.DataFrame,
    window: int | None = None,
) -> bool:
    """Verifica que a normalização não introduziu look-ahead bias.

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
    # Consistente com normalize_features: ignora colunas raw OHLCV
    check_cols = [c for c in numeric_cols if c not in _BLOCKLIST_RAW_COLS]

    # Testa um subconjunto de pontos para performance
    n_tests = min(50, len(original))
    test_indices = np.linspace(window, len(original) - 1, n_tests, dtype=int)

    for col in check_cols:
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

"""
Diferenciação Fracionária (FFD) — TradeSystem5000.

Este módulo implementa a técnica de Diferenciação Fracionária (FracDiff) com
janela de largura fixa (Fixed-Width Window). O objetivo é tornar a série
financeira estacionária preservando o máximo possível de "memória" (correlação
serial de longo prazo).

Funcionalidades:
- **frac_diff_ffd**: Aplicação da transformação FFD via convolução.
- **find_min_d**: Busca automatizada do parâmetro 'd' mínimo via teste ADF.
- **get_weights_ffd**: Cálculo dos pesos iterativos da diferenciação.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulo 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from numba import njit
from statsmodels.tsa.stattools import adfuller

from config.settings import feature_config


# ---------------------------------------------------------------------------
# Pesos FFD
# ---------------------------------------------------------------------------
def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Calcula os pesos da diferenciação fracionária com janela fixa (FFD).

    Os pesos são expandidos até que ``|w_k| < threshold``.

    Parameters
    ----------
    d : float
        Ordem de diferenciação (0 < d < 1 tipicamente).
    threshold : float
        Valor mínimo absoluto de peso para inclusão.

    Returns
    -------
    np.ndarray
        Array de pesos (do mais recente ao mais antigo).
    """
    weights = [1.0]
    k = 1

    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1

    weights = np.array(weights[::-1], dtype=np.float64)
    logger.debug("FFD pesos: d={}, threshold={}, tamanho={}", d, threshold, len(weights))
    return weights


# ---------------------------------------------------------------------------
# Aplicação FFD
# ---------------------------------------------------------------------------
def frac_diff_ffd(
    series: pd.Series,
    d: float | None = None,
    threshold: float | None = None,
) -> pd.Series:
    """
    Aplica diferenciação fracionária FFD a uma série temporal.

    Usa convolução point-in-time (sem look-ahead). Os primeiros
    ``len(weights) - 1`` valores são descartados (NaN).

    Parameters
    ----------
    series : pd.Series
        Série de preços (tipicamente log-preços ou preços).
    d : float, optional
        Ordem de diferenciação. Default: config.
    threshold : float, optional
        Corte de pesos. Default: config.

    Returns
    -------
    pd.Series
        Série diferenciada fracionariamente, com mesmo índice.
    """
    if d is None:
        d = feature_config.ffd_d
    if threshold is None:
        threshold = feature_config.ffd_threshold

    weights = get_weights_ffd(d, threshold)
    width = len(weights)
    values = series.values.astype(np.float64)

    logger.info("Aplicando FFD: d={}, janela_pesos={}, len_série={}", d, width, len(values))

    output = pd.Series(np.nan, index=series.index, name=series.name, dtype=np.float64)

    if width > len(values):
        logger.warning(
            "FFD: janela de pesos ({}) maior que série ({}). Retornando NaN.",
            width,
            len(values),
        )
        return output

    result = _apply_ffd_kernel(values, weights)
    output.iloc[width - 1 :] = result

    return output


@njit
def _apply_ffd_kernel(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Kernel numba para convolução FFD point-in-time."""
    width = len(weights)
    n = len(values) - width + 1
    result = np.empty(n, dtype=np.float64)

    for i in range(n):
        result[i] = 0.0
        for j in range(width):
            result[i] += weights[j] * values[i + j]

    return result


# ---------------------------------------------------------------------------
# Busca automática do d mínimo
# ---------------------------------------------------------------------------
def find_min_d(
    series: pd.Series,
    d_range: np.ndarray | None = None,
    threshold: float | None = None,
    adf_pvalue: float | None = None,
) -> float:
    """
    Encontra o menor ``d`` que torna a série estacionária (teste ADF).

    Varre valores de ``d`` em ordem crescente e retorna o primeiro
    cuja série diferenciada tem p-value ADF ≤ ``adf_pvalue``.

    Parameters
    ----------
    series : pd.Series
        Série original (não estacionária).
    d_range : np.ndarray, optional
        Valores de d a testar. Default: ``np.arange(0, 1.05, 0.05)``.
    threshold : float, optional
        Corte de pesos FFD. Default: config.
    adf_pvalue : float, optional
        P-value alvo para estacionaridade. Default: config.

    Returns
    -------
    float
        Menor ``d`` que atinge estacionaridade. Retorna 1.0 se nenhum ``d``
        fracionário for suficiente.
    """
    if d_range is None:
        d_range = np.arange(0.05, 1.05, 0.05)
    if threshold is None:
        threshold = feature_config.ffd_threshold
    if adf_pvalue is None:
        adf_pvalue = feature_config.ffd_adf_pvalue

    logger.info(
        "Buscando d mínimo: range=[{:.2f}, {:.2f}], adf_target={}",
        d_range[0],
        d_range[-1],
        adf_pvalue,
    )

    for d in d_range:
        diff = frac_diff_ffd(series, d=d, threshold=threshold).dropna()

        if len(diff) < 20:
            logger.debug("d={:.2f}: amostras insuficientes ({})", d, len(diff))
            continue

        try:
            adf_stat, pval, *_ = adfuller(diff, maxlag=1, regression="c", autolag=None)
        except Exception:
            logger.debug("d={:.2f}: ADF falhou", d)
            continue

        logger.debug("d={:.2f}: ADF stat={:.4f}, p-value={:.4f}", d, adf_stat, pval)

        if pval <= adf_pvalue:
            logger.success("d mínimo encontrado: {:.2f} (p-value={:.4f})", d, pval)
            return float(d)

    logger.warning("Nenhum d fracionário atingiu estacionaridade. Retornando d=1.0")
    return 1.0

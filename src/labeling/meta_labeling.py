"""
3.4 — Meta-Labeling.

Gera rótulos binários para o modelo secundário (meta-model):
- ``1``: o sinal do Alpha atingiu Take Profit (acertou)
- ``0``: o sinal do Alpha atingiu Stop Loss ou tempo expirou (errou)

O meta-model aprende a filtrar os sinais do Alpha, prevendo quais
sinais têm maior probabilidade de ser "verdadeiros positivos".

Referência: López de Prado, *Advances in Financial Machine Learning*, Cap. 3.6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import labeling_config


# ---------------------------------------------------------------------------
# Geração de Meta-Labels
# ---------------------------------------------------------------------------
def get_meta_labels(
    triple_barrier_result: pd.DataFrame,
    min_return: float | None = None,
) -> pd.Series:
    """
    Converte o resultado da Tripla Barreira em rótulos binários para meta-labeling.

    O meta-label indica se o Alpha Model *acertou* a direção:
    - ``1``: o trade do Alpha foi lucrativo (TP atingido ou retorno positivo no tempo)
    - ``0``: o trade do Alpha falhou (SL atingido ou retorno negativo/neutro no tempo)

    Parameters
    ----------
    triple_barrier_result : pd.DataFrame
        Output de ``triple_barrier.get_labels()`` ou ``apply_triple_barrier()``.
        Deve conter colunas: ``ret``, ``side``, ``barrier_type`` (ou ``label``).
    min_return : float, optional
        Retorno mínimo para considerar sucesso. Default: config.

    Returns
    -------
    pd.Series
        Série binária {0, 1} com mesmo índice dos eventos.
        Nome: ``meta_label``.
    """
    if min_return is None:
        min_return = labeling_config.min_return

    meta = pd.Series(0, index=triple_barrier_result.index, dtype=np.int8, name="meta_label")

    # Se tem coluna 'label' (output de get_labels)
    if "label" in triple_barrier_result.columns:
        meta[triple_barrier_result["label"] == 1] = 1
    # Se tem coluna 'ret' (output bruto)
    elif "ret" in triple_barrier_result.columns:
        meta[triple_barrier_result["ret"] > min_return] = 1

    n_pos = (meta == 1).sum()
    n_neg = (meta == 0).sum()
    ratio = n_pos / max(1, n_pos + n_neg)

    logger.info(
        "Meta-labels: {} positivos, {} negativos (ratio={:.2%})",
        n_pos,
        n_neg,
        ratio,
    )

    return meta


# ---------------------------------------------------------------------------
# Dataset completo de treinamento
# ---------------------------------------------------------------------------
def build_training_dataset(
    features: pd.DataFrame,
    triple_barrier_result: pd.DataFrame,
    include_side: bool = True,
    include_return_info: bool = False,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Monta o dataset final de treinamento combinando features + meta-labels.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame com features calculadas (output da Fase 2).
    triple_barrier_result : pd.DataFrame
        Resultado da Tripla Barreira (output de ``get_labels``).
    include_side : bool
        Se True, inclui coluna ``side`` (direção do Alpha) como feature.
    include_return_info : bool
        Se True, inclui ``ret`` e ``barrier_type`` para análise (não como feature).
    dropna : bool
        Se True, remove linhas com NaN.

    Returns
    -------
    pd.DataFrame
        Dataset pronto para treinamento com:
        - Colunas de features
        - Coluna ``meta_label`` (target)
        - Opcionalmente ``side``, ``ret``, ``barrier_type``
    """
    # Alinha features com os eventos da tripla barreira
    event_idx = triple_barrier_result.index
    common_idx = event_idx.intersection(features.index)

    if len(common_idx) < len(event_idx):
        logger.debug(
            "Alinhamento features/eventos: {} de {} eventos com features",
            len(common_idx),
            len(event_idx),
        )

    # Filtra ambos para índice comum
    feat_aligned = features.loc[common_idx].copy()
    tb_aligned = triple_barrier_result.loc[common_idx]

    # Meta-label
    meta = get_meta_labels(tb_aligned)
    feat_aligned["meta_label"] = meta

    # Side como feature
    if include_side and "side" in tb_aligned.columns:
        feat_aligned["side"] = tb_aligned["side"]

    # Info de retorno (para análise, não treino)
    if include_return_info:
        if "ret" in tb_aligned.columns:
            feat_aligned["_ret"] = tb_aligned["ret"]
        if "barrier_type" in tb_aligned.columns:
            feat_aligned["_barrier_type"] = tb_aligned["barrier_type"]

    if dropna:
        before = len(feat_aligned)
        feat_aligned = feat_aligned.dropna()
        dropped = before - len(feat_aligned)
        if dropped > 0:
            logger.debug("NaN removidos: {} linhas", dropped)

    logger.success(
        "Dataset de treinamento: {} amostras, {} features, meta_label ratio={:.2%}",
        len(feat_aligned),
        len(feat_aligned.columns) - 1,  # exclui meta_label
        feat_aligned["meta_label"].mean() if len(feat_aligned) > 0 else 0.0,
    )

    return feat_aligned


# ---------------------------------------------------------------------------
# Análise de meta-labels
# ---------------------------------------------------------------------------
def meta_label_analysis(
    triple_barrier_result: pd.DataFrame,
) -> dict:
    """
    Retorna estatísticas descritivas do resultado da meta-rotulagem.

    Parameters
    ----------
    triple_barrier_result : pd.DataFrame
        Resultado da Tripla Barreira.

    Returns
    -------
    dict
        Dicionário com métricas descritivas.
    """
    meta = get_meta_labels(triple_barrier_result)

    n_total = len(meta)
    n_pos = (meta == 1).sum()
    n_neg = (meta == 0).sum()

    stats = {
        "total_events": n_total,
        "positive_labels": int(n_pos),
        "negative_labels": int(n_neg),
        "positive_ratio": n_pos / max(1, n_total),
        "negative_ratio": n_neg / max(1, n_total),
    }

    # Breakdown por tipo de barreira
    if "barrier_type" in triple_barrier_result.columns:
        bt_counts = triple_barrier_result["barrier_type"].value_counts().to_dict()
        stats["barrier_breakdown"] = bt_counts

    # Estatísticas de retorno
    if "ret" in triple_barrier_result.columns:
        rets = triple_barrier_result["ret"]
        stats["mean_return"] = float(rets.mean())
        stats["median_return"] = float(rets.median())
        stats["std_return"] = float(rets.std())
        stats["mean_return_positive"] = float(rets[meta == 1].mean()) if n_pos > 0 else 0.0
        stats["mean_return_negative"] = float(rets[meta == 0].mean()) if n_neg > 0 else 0.0

    logger.info("Meta-label analysis: {}", stats)
    return stats

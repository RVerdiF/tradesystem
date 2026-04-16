"""Meta-Labeling (Rotulagem Secundária) — TradeSystem5000.

Este módulo implementa a técnica de Meta-Labeling para treinar um modelo
secundário (Meta-Modelo) capaz de filtrar os sinais do modelo de Alpha.

O Meta-Modelo aprende a prever a probabilidade de sucesso de uma operação,
atuando como um filtro de falsos positivos e permitindo o dimensionamento
ótimo das apostas (Bet Sizing).

Funcionalidades:
- **get_meta_labels**: Geração de labels binárias {0, 1} baseadas no sucesso do Alpha.
- **build_training_dataset**: Alinhamento de features e labels para o treinamento.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulo 3.6.
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
    """Rotula o resultado da Tripla Barreira em rótulos binários para meta-labeling.

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

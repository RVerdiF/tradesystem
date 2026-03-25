"""
4.1 — Purga e Embargo de Dados (Purging & Embargoing).

Evita o vazamento de dados (data leakage) em séries temporais financeiras
durante a validação cruzada.
- **Purga**: remove amostras no conjunto de treino cujos períodos de avaliação
  se sobrepõem aos blocos de teste.
- **Embargo**: remove um período logo após o bloco de teste para evitar viés
  causado por autocorrelação serial.

Referência: López de Prado, *Advances in Financial Machine Learning*, Cap. 7.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from config.settings import ml_config


# ---------------------------------------------------------------------------
# Purga (Purging)
# ---------------------------------------------------------------------------
def get_train_times(
    t1: pd.Series,
    test_times: pd.Series,
) -> pd.Series:
    """
    Realiza a purga removendo sobreposições entre treino e teste.

    Retorna os tempos de treino válidos que NÃO sobrepõem o intervalo
    do conjunto de teste.

    Parameters
    ----------
    t1 : pd.Series
        Timestamps do fim de cada trade (barreira vertical, TP ou SL).
        Índice = início do trade.
    test_times : pd.Series
        Série contendo os tempos de início (índice) e fim (valores) do
        bloco de teste atual.

    Returns
    -------
    pd.Series
        Subconjunto de `t1` contendo apenas instâncias de treino que não sofrem
        vazamento de dados para o `test_times`.
    """
    train_times = t1.copy(deep=True)

    # Avalia a sobreposição de cada período de teste
    for start_test, end_test in test_times.items():
        # Sobreposição tipo 1:
        # Treino começa ANTES ou JUNTO do teste, mas termina DENTRO ou DEPOIS.
        # Condição: t0_train <= t0_test e t1_train > t0_test
        cond1 = (train_times.index <= start_test) & (train_times > start_test)

        # Sobreposição tipo 2:
        # Treino começa DENTRO do teste.
        # Condição: t0_train >= t0_test e t0_train <= t1_test
        cond2 = (train_times.index >= start_test) & (train_times.index <= end_test)

        # Sobreposição tipo 3:
        # Treino engloba todo o teste. (Já capturado na cond1)

        overlap = cond1 | cond2
        train_times = train_times.loc[~overlap]

    return train_times


# ---------------------------------------------------------------------------
# Embargo
# ---------------------------------------------------------------------------
def apply_embargo(
    t1: pd.Series,
    test_times: pd.Series,
    pct_embargo: float | None = None,
    step: pd.Timedelta | None = None,
) -> pd.Series:
    """
    Aplica o embargo após o conjunto de teste.

    O embargo remove amostras de treino que iniciam imediatamente após
    o período de teste para prevenir vazamento via autocorrelação.

    Parameters
    ----------
    t1 : pd.Series
        Série dos tempos de evento (purificada por `get_train_times`).
    test_times : pd.Series
        Tempos de início/fim do bloco de teste atual.
    pct_embargo : float, optional
        Fração das barras totais a serem embargadas. Padrão vem da config.
    step : pd.Timedelta, optional
        Tempo médio equivalente a pct_embargo. Se não provido, será
        calculado assumindo distribuição uniforme baseada em `t1`.

    Returns
    -------
    pd.Series
        `t1` ajustado sem o período de embargo.
    """
    if pct_embargo is None:
        pct_embargo = ml_config.embargo_pct

    if pct_embargo <= 0:
        return t1

    train_times = t1.copy(deep=True)

    # Estimativa de tempo do embargo
    if step is None and len(t1) > 1:
        total_time = t1.index[-1] - t1.index[0]
        embargo_duration = total_time * pct_embargo
    elif step is not None:
        embargo_duration = step
    else:
        # Fallback curto se não puder ser estimado
        embargo_duration = pd.Timedelta(seconds=1)

    for _, end_test in test_times.items():
        embargo_end = end_test + embargo_duration

        # Remove todas as amostras de treino que começam durante o embargo
        # Condição: t0_train > t1_test e t0_train <= t1_test + embargo
        overlap = (train_times.index > end_test) & (train_times.index <= embargo_end)
        train_times = train_times.loc[~overlap]

    return train_times


# ---------------------------------------------------------------------------
# Pipeline completo Purged + Embargo
# ---------------------------------------------------------------------------
def purge_and_embargo(
    t1: pd.Series,
    test_times: pd.Series,
    pct_embargo: float | None = None,
) -> pd.Series:
    """
    Executa purga e embargo de uma só vez.

    Parameters
    ----------
    t1 : pd.Series
        Série temporal completa com início/fim dos trades de todo o dataset.
    test_times : pd.Series
        Início/fim do bloco de teste específico de uma iteração de CV.
    pct_embargo : float, optional
        Tamanho do embargo.

    Returns
    -------
    pd.Series
        Amostras válidas do conjunto de treino após purga e embargo.
    """
    # 1. Purga
    train_purged = get_train_times(t1, test_times)

    # 2. Embargo
    train_valid = apply_embargo(train_purged, test_times, pct_embargo)

    n_dropped = len(t1) - len(train_valid) - len(test_times)
    logger.debug(
        "Purge/Embargo: excluídos {} de {} ({} teste)", n_dropped, len(t1), len(test_times)
    )

    return train_valid

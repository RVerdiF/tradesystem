"""
4.2 — Purged K-Fold Cross Validation.

Estende o K-Fold tradicional do scikit-learn para suportar séries temporais
financeiras usando Purging e Embargoing, evitando o vazamento (data leakage)
de informações futuras no treinamento.

Referência: López de Prado, *Advances in Financial Machine Learning*, Cap. 7.4.
"""

from __future__ import annotations

import collections.abc
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold
from loguru import logger

from src.modeling.purge_embargo import apply_embargo, get_train_times
from config.settings import ml_config


class PurgedKFold(_BaseKFold):
    """
    K-Fold Cross Validation com Purga e Embargo para séries temporais.

    Garante que as observações de treinamento não se sobreponham 
    (no tempo) com o conjunto de teste (purga) e que haja uma lacuna
    após o conjunto de teste para evitar autocorrelação (embargo).

    Parameters
    ----------
    n_splits : int, optional
        Número de folds. Padrão vem de ml_config.
    samples_info : pd.Series
        Série contendo o índice (data de início do evento) e os valores
        (data de fim do evento), correspondendo aos trades ou features.
    pct_embargo : float, optional
        Fração da amostra usada para o embargo pós-teste.
    """

    def __init__(
        self,
        samples_info: pd.Series,
        n_splits: int | None = None,
        pct_embargo: float | None = None,
    ) -> None:
        if not isinstance(samples_info, pd.Series):
            raise ValueError("samples_info deve ser uma pd.Series (index=t0, values=t1)")

        n_splits = n_splits or ml_config.cv_splits
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)

        self.samples_info = samples_info
        self.pct_embargo = pct_embargo if pct_embargo is not None else ml_config.embargo_pct

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> collections.abc.Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Gera os índices para treino e teste para cada fold.

        Parameters
        ----------
        X : pd.DataFrame ou np.ndarray
            Features.
        y : pd.Series ou np.ndarray, optional
            Apenas para compatibilidade com scikit-learn.
        groups : array-like, optional
            Sempre ignorado (não faz sentido aqui).

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            (train_indices, test_indices)
        """
        if isinstance(X, pd.DataFrame):
            indices = np.arange(X.shape[0])
        else:
            indices = np.arange(len(X))

        if len(self.samples_info) != len(indices):
            raise ValueError("O tamanho de samples_info e X devem ser iguais.")

        # test_bounds divide o array em n_splits intervalos de teste
        test_bounds = [(t[0], t[-1] + 1) for t in np.array_split(indices, self.n_splits)]

        for i, (start, end) in enumerate(test_bounds):
            test_indices = indices[start:end]

            # Encontra o tempo correspondente ao bloco de teste usando samples_info
            # t0 (início do split de teste) até t1 máximo (final do teste)
            test_times = pd.Series(
                index=[self.samples_info.index[start]],
                data=[self.samples_info.iloc[start:end].max()],
            )

            # train_indices temporários: tudo fora do split de teste
            train_indices_temp = np.concatenate((indices[:start], indices[end:]))

            # Filtramos a info de treino para os candidatos a treino
            train_info = self.samples_info.iloc[train_indices_temp]

            # 1. Purga
            train_purged = get_train_times(train_info, test_times)

            # 2. Embargo
            train_valid = apply_embargo(train_purged, test_times, self.pct_embargo)

            # Filtra os índices originais para manter apenas os válidos
            valid_train_idx = []
            for t_idx in train_valid.index:
                # Onde o t_idx original está nos índices gerais
                loc = self.samples_info.index.get_loc(t_idx)
                # se houver duplicatas no evento t0, get_loc retorna slice/array
                if isinstance(loc, slice):
                    valid_train_idx.extend(range(loc.start, loc.stop))
                elif isinstance(loc, np.ndarray):
                    valid_train_idx.extend(np.where(loc)[0])
                else:
                    valid_train_idx.append(loc)

            valid_train_idx = np.array(valid_train_idx)

            logger.debug(
                "Fold {}/{} | Train: {} | Test: {}",
                i + 1,
                self.n_splits,
                len(valid_train_idx),
                len(test_indices),
            )

            yield valid_train_idx, test_indices


# ---------------------------------------------------------------------------
# Cross-Validation Utility
# ---------------------------------------------------------------------------
def cv_score(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | None,
    scoring: str | Mapping = "accuracy",
    samples_info: pd.Series | None = None,
    n_splits: int | None = None,
) -> pd.DataFrame:
    """
    Utilitário para rodar PurgedKFold facilmente.
    """
    from sklearn.model_selection import cross_validate

    if samples_info is None:
        raise ValueError("samples_info (t0 -> t1) é obrigatório para Purged CV.")

    cv = PurgedKFold(samples_info=samples_info, n_splits=n_splits)

    fit_params = {}
    if sample_weight is not None:
        fit_params["sample_weight"] = sample_weight.values

    scores = cross_validate(
        clf,
        X.values,
        y.values,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
    )

    df_scores = pd.DataFrame(scores)
    logger.info("CV médio: {:.4f} ± {:.4f}", df_scores["test_score"].mean(), df_scores["test_score"].std())
    return df_scores

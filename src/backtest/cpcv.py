"""
5.1 — Combinatorial Purged Cross-Validation (CPCV).

Gera múltiplos caminhos de backtest combinatórios a partir de N grupos,
cada caminho sendo uma sequência contígua de blocos teste enquanto os
demais servem como treino — com purga e embargo aplicados.

Permite avaliar a distribuição de performance do modelo sob inúmeros
cenários históricos não-lineares, reduzindo o risco de overfitting e
o viés de seleção de caminho.

Referência: López de Prado, *Advances in Financial Machine Learning*, Cap. 12.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from loguru import logger

from src.modeling.purge_embargo import get_train_times, apply_embargo


# ---------------------------------------------------------------------------
# Combinatorial Purged Cross-Validation
# ---------------------------------------------------------------------------
class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation.

    Divide a série em ``n_groups`` blocos ordenados no tempo e gera
    todos os caminhos onde ``n_test_groups`` blocos consecutivos formam
    o conjunto de teste (com purga + embargo).

    Parameters
    ----------
    n_groups : int
        Número total de blocos (grupos) para dividir a série.
    n_test_groups : int
        Quantos blocos compõem cada conjunto de teste.
    samples_info : pd.Series
        Série t0 (index) → t1 (values) com os tempos dos eventos.
    pct_embargo : float
        Fração de embargo pós-teste.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        samples_info: pd.Series | None = None,
        pct_embargo: float = 0.01,
    ) -> None:
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.samples_info = samples_info
        self.pct_embargo = pct_embargo

    @property
    def n_paths(self) -> int:
        """Número total de caminhos (combinações) gerados."""
        from math import comb
        return comb(self.n_groups, self.n_test_groups)

    def get_combinations(self) -> list[tuple[int, ...]]:
        """Retorna todas as combinações de blocos de teste."""
        return list(combinations(range(self.n_groups), self.n_test_groups))

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Gera todos os splits combinatórios com purga e embargo.

        Parameters
        ----------
        X : pd.DataFrame ou np.ndarray
            Features (usado apenas para determinar tamanho).
        y : ignored

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            Lista de (train_indices, test_indices) para cada combinação.
        """
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        indices = np.arange(n_samples)

        # Divide em n_groups blocos contíguos
        groups = np.array_split(indices, self.n_groups)
        combos = self.get_combinations()

        splits = []
        for combo in combos:
            # Índices de teste: blocos selecionados na combinação
            test_idx = np.concatenate([groups[g] for g in combo])

            # Índices candidatos a treino: tudo que não é teste
            train_candidates = np.setdiff1d(indices, test_idx)

            # Aplica purga e embargo se temos samples_info
            if self.samples_info is not None and len(self.samples_info) == n_samples:
                test_info = self.samples_info.iloc[test_idx]
                test_times = pd.Series(
                    index=[test_info.index.min()],
                    data=[test_info.max()],
                )

                train_info = self.samples_info.iloc[train_candidates]
                train_purged = get_train_times(train_info, test_times)
                train_valid = apply_embargo(train_purged, test_times, self.pct_embargo)

                # Converte de volta para índices numéricos
                valid_idx = []
                for ts in train_valid.index:
                    loc = self.samples_info.index.get_loc(ts)
                    if isinstance(loc, slice):
                        valid_idx.extend(range(loc.start, loc.stop))
                    elif isinstance(loc, np.ndarray):
                        valid_idx.extend(np.where(loc)[0])
                    else:
                        valid_idx.append(loc)
                train_idx = np.array(valid_idx)
            else:
                train_idx = train_candidates

            splits.append((train_idx, test_idx))

        logger.info(
            "CPCV: {} caminhos | {} grupos | {} teste por caminho",
            len(splits),
            self.n_groups,
            self.n_test_groups,
        )
        return splits

    def backtest_paths(
        self,
        predictions: dict[int, pd.Series],
    ) -> pd.DataFrame:
        """
        Reconstrói os caminhos de backtest a partir das previsões por fold.

        Parameters
        ----------
        predictions : dict[int, pd.Series]
            Dicionário {combo_index: previsões_no_teste} para cada split.

        Returns
        -------
        pd.DataFrame
            DataFrame com colunas = caminhos e valores = retornos previstos.
        """
        paths = {}
        for combo_idx, preds in predictions.items():
            paths[f"path_{combo_idx}"] = preds

        result = pd.DataFrame(paths)
        logger.info(
            "Backtest paths: {} caminhos, {} observações máx",
            result.shape[1],
            result.count().max(),
        )
        return result

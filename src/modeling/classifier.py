"""
4.3 — Classificador Secundário (Meta-Model).

O meta-model não prevê a direção do mercado (o Alpha Model já o fez), mas sim
o "tamanho do acerto" ou a probabilidade de o trade ser bem sucedido.
Usa as features do momento da entrada (T0) e os meta-labels (0, 1) produzidos
pelo módulo da Tripla Barreira.

Random Forest ou XGBoost são adequados por lidarem bem com não-linearidades e outliers.

Referência: López de Prado, *Advances in Financial Machine Learning*, Cap. 3 / Cap. 7.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class MetaClassifier(BaseEstimator, ClassifierMixin):
    """
    Meta-Classificador Secundário para filtrar sinais do modelo primário.

    Wraps um XGBoost Classifier ou RandomForestClassifier.

    Parameters
    ----------
    n_estimators : int
        Número de árvores na floresta.
    max_depth : int, optional
        Profundidade máxima. Regularização severa é importante em finanças 
        (overfitting é o maior vilão).
    class_weight : str or dict, default="balanced"
        Essencial já que a classe de sucesso (1) pode ser muito desbalanceada.
    n_jobs : int, default=-1
        Processamento paralelo.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 5,
        class_weight: str | dict | None = "balanced",
        n_jobs: int = -1,
        random_state: int | None = 42,
        use_xgboost: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_xgboost = use_xgboost and HAS_XGB

        if self.use_xgboost:
            logger.info("Usando XGBoost Classifier para Meta-Model.")
            # XGBoost não aceita "balanced" diretamente no class_weight, 
            # mas podemos setar scale_pos_weight posteriormente ou usar parâmetros padrão.
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                eval_metric="logloss",
            )
        else:
            logger.info("Usando Random Forest para Meta-Model.")
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight="balanced_subsample" if self.class_weight == "balanced" else self.class_weight,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> MetaClassifier:
        """
        Treina o classificador Secundário.

        Parameters
        ----------
        X : DataFrame (features)
        y : Series (labels: 0 ou 1)
        sample_weight : Series_like, optional
            Pesos para cada amostra (p. ex., retorno absoluto para dar
            mais peso a predições de trades maiores).
        """
        logger.info("Treinando MetaClassifier com {} amostras e {} features", min(X.shape), X.shape[1] if len(X.shape)>1 else 0)

        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted_ = True

        # Métricas de treino só pra debug rápido
        train_proba = self.model.predict_proba(X)[:, 1]
        try:
            auc = roc_auc_score(y, train_proba)
            logger.debug("AUC no treino: {:.4f} (Overfit?)", auc)
        except ValueError:
            pass  # Somente 1 classe no y

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predição discreta {0, 1}."""
        if not self.is_fitted_:
            raise RuntimeError("O modelo precisa ser fitado antes do uso.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Retorna as probabilidades.

        A coluna 1 é a probabilidade do trade do Alpha Model ser lucrativo, 
        que será a principal entrada para o dimensionamento de posição (Bet Sizing).
        """
        if not self.is_fitted_:
            raise RuntimeError("O modelo precisa ser fitado antes do uso.")
        return self.model.predict_proba(X)

    def feature_importances(self, feature_names: list[str] | None = None) -> pd.Series:
        """
        Extrai importância das features extraída do Random Forest (MDI).

        Nota: Feature Importance por MDI em finanças pode ser enganosa se houver
        correlação ou substituição. Para aprofundar, veja Cap. 8 de AFML (MDA/CFI).
        """
        if not self.is_fitted_:
            raise RuntimeError("O modelo não está ajustado.")

        importances = self.model.feature_importances_
        if feature_names is not None:
            return pd.Series(importances, index=feature_names).sort_values(ascending=False)
        return pd.Series(importances).sort_values(ascending=False)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Relatório de avaliação rápido"""
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        try:
            auc = roc_auc_score(y_test, y_prob)
            report["auc"] = auc
        except ValueError:
            report["auc"] = np.nan

        logger.info(
            "Evaluation: F1(1)={:.2f} | Acc={:.2f} | AUC={:.2f}",
            report.get("1", {}).get("f1-score", 0),
            report.get("accuracy", 0),
            report.get("auc", 0),
        )

        return report

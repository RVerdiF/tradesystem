"""
Avaliação de Features e Auditoria de Modelos — TradeSystem5000.

Este módulo implementa técnicas avançadas de explicabilidade e auditoria de
features para validar a utilidade das variáveis e evitar overfitting.

Métodos:
- **SHAP (Shapley Additive Explanations)**: Atribuição de importância baseada em teoria dos jogos.
- **MDA (Mean Decrease Accuracy)**: Importância por permutação (redução de acurácia).

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulo 8.
"""

import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.metrics import roc_auc_score


def evaluate_features_shap(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Calcula a importância das features via SHAP (Shapley Additive Explanations).

    Utiliza o TreeExplainer para extrair as contribuições médias absolutas de cada
    feature para a predição do modelo.

    Parameters
    ----------
    model : MetaClassifier
        Modelo treinado (ou qualquer modelo compatível com SHAP TreeExplainer).
    X : pd.DataFrame
        Conjunto de features (X_train ou X_test).
    y : pd.Series
        Alvos (utilizados para contexto, opcional em alguns explicadores).

    Returns
    -------
    pd.DataFrame
        DataFrame ordenado com colunas ['col_name', 'feature_importance_vals'].
    """
    logger.info("Iniciando avaliação explicativa SHAP...")

    try:
        # SHAP TreeExplainer para XGBoost/RandomForest
        # Acessamos o objeto interno 'model' do wrapper MetaClassifier
        inner_model = getattr(model, "model", model)
        explainer = shap.TreeExplainer(inner_model)
        shap_values = explainer.shap_values(X)

        # Para classificação binária, o formato pode variar entre versões do SHAP e modelos
        if isinstance(shap_values, list):
            # No RandomForest do sklearn, retorna lista [prob_0, prob_1]
            vals = np.abs(shap_values[1]).mean(0)
        elif len(shap_values.shape) == 3:
            # Versões novas do SHAP com XGBoost podem retornar (amostras, features, classes)
            vals = np.abs(shap_values[:, :, 1]).mean(0)
        else:
            # XGBoost padrão geralmente retorna (amostras, features)
            vals = np.abs(shap_values).mean(0)

        feature_importance = pd.DataFrame(
            list(zip(X.columns, vals)), columns=["col_name", "feature_importance_vals"]
        )
        feature_importance.sort_values(
            by=["feature_importance_vals"], ascending=False, inplace=True
        )

        logger.info("Resumo SHAP (Importância Média Absoluta - Top 10):")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f" - {row['col_name']}: {row['feature_importance_vals']:.6f}")

        return feature_importance

    except Exception as e:
        logger.error(f"Erro ao calcular SHAP: {e}")
        return pd.DataFrame()


def evaluate_features_mda(
    model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5
) -> pd.Series:
    """
    Calcula a importância via Mean Decrease Accuracy (MDA) por Permutação.

    Mede a queda na performance do modelo (ROC-AUC) ao embaralhar os valores
    de cada feature individualmente. Features que causam maior queda são
    consideradas mais importantes.

    Parameters
    ----------
    model : MetaClassifier
        Modelo treinado.
    X : pd.DataFrame
        Conjunto de features.
    y : pd.Series
        Alvos verdadeiros.
    n_repeats : int, optional
        Número de permutações por feature para estabilizar a estimativa. Default: 5.

    Returns
    -------
    pd.Series
        Série com a queda média no ROC-AUC por feature.
    """
    logger.info("Iniciando auditoria matemática (MDA) via Permutação...")

    try:
        baseline_prob = model.predict_proba(X)[:, 1]
        baseline_auc = roc_auc_score(y, baseline_prob)

        mda_scores = {}

        for col in X.columns:
            scores = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                X_perm[col] = np.random.permutation(X_perm[col].values)
                perm_prob = model.predict_proba(X_perm)[:, 1]
                perm_auc = roc_auc_score(y, perm_prob)
                # MDA = Perda de performance ao embaralhar a feature
                scores.append(baseline_auc - perm_auc)

            mda_scores[col] = np.mean(scores)

        mda_df = pd.Series(mda_scores).sort_values(ascending=False)

        logger.info("Relatório MDA (Queda média no ROC-AUC):")
        for col, val in mda_df.items():
            if val > 0.0001:
                logger.info(f" - {col}: {val:.6f} (Contribuição Positiva)")
            elif val > -0.0001:
                logger.warning(f" - {col}: {val:.6f} (Insignificante/Neutra)")
            else:
                logger.error(
                    f" - {col}: {val:.6f} (RISCO: Redundante ou Ruído - MDA Negativo)"
                )

        return mda_df

    except Exception as e:
        logger.error(f"Erro ao calcular MDA: {e}")
        return pd.Series()

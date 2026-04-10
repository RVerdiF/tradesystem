"""
Testes para a auditoria de features (SHAP e MDA).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier
from src.modeling.feature_evaluation import evaluate_features_shap, evaluate_features_mda

# ---------------------------------------------------------------------------
# Fixtures e Mocks
# ---------------------------------------------------------------------------
class DummyMetaClassifierMock:
    """Mock do MetaClassifier para simular o comportamento no `predict_proba` e ter um `model` interno."""
    
    def __init__(self, random_state=42):
        self.model = DecisionTreeClassifier(max_depth=3, random_state=random_state)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)


@pytest.fixture
def synthetic_data():
    """Gera dados determinísticos para teste: 1 feature preditiva e 2 ruidosas."""
    np.random.seed(42)
    n = 100
    
    # Feature 1 é exata
    f1 = np.random.randn(n)
    y = (f1 > 0).astype(int)
    
    # Features 2 e 3 são ruído
    f2 = np.random.randn(n)
    f3 = np.random.randn(n)
    
    X = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3})
    y_series = pd.Series(y)
    
    return X, y_series


@pytest.fixture
def trained_mock_model(synthetic_data):
    """Retorna um modelo mock treinado."""
    X, y = synthetic_data
    mock = DummyMetaClassifierMock()
    mock.fit(X, y)
    return mock


# ---------------------------------------------------------------------------
# Testes - SHAP
# ---------------------------------------------------------------------------
class TestEvaluateFeaturesSHAP:
    def test_shap_dataframe_format(self, trained_mock_model, synthetic_data):
        """Verifica se retorna DataFrame ordenado e com colunas corretas."""
        X, y = synthetic_data
        result = evaluate_features_shap(trained_mock_model, X, y)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col_name", "feature_importance_vals"]
        assert len(result) == 3
        
        # A importância maior deve ser da feature 1 (preditiva)
        top_feature = result.iloc[0]["col_name"]
        assert top_feature == "f1", f"Feature preditiva não ficou no topo. Topo foi {top_feature}"

    def test_shap_graceful_error_handling(self):
        """Se passar um modelo inválido, deve capturar a exceção e retornar df vazio."""
        class InvalidModel:
            pass
            
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        
        result = evaluate_features_shap(InvalidModel(), X, y)
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ---------------------------------------------------------------------------
# Testes - MDA (Mean Decrease Accuracy)
# ---------------------------------------------------------------------------
class TestEvaluateFeaturesMDA:
    def test_mda_series_format(self, trained_mock_model, synthetic_data):
        """Verifica formato da Series retornada pela permutação."""
        X, y = synthetic_data
        result = evaluate_features_mda(trained_mock_model, X, y, n_repeats=3)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert list(result.index).count("f1") == 1
        
        # f1 deve ter a maior queda de performance (MDA mais alto)
        # Permutar algo que tem 100% corrulação com Y deve destruir AUC
        assert result.idxmax() == "f1"
        assert result["f1"] > 0.1 # Deve haver uma queda notável no AUC
        
    def test_mda_graceful_error_handling(self):
        """Se der erro no ROC AUC (ex: y só tem 1 classe), deve tratar a exceção e retornar vazio."""
        X = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
        # Força erro: y_true tendo apenas classe 0 para AUC (causa ValueError em roc_auc_score)
        y = pd.Series([0, 0])
        
        mock = DummyMetaClassifierMock()
        mock.fit(X, y)
        
        result = evaluate_features_mda(mock, X, y, n_repeats=1)
        assert isinstance(result, pd.Series)
        assert result.empty

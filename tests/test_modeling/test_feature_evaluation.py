from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.modeling.feature_evaluation import evaluate_features_mda, evaluate_features_shap


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


class TestEvaluateFeaturesSHAP:
    def test_shap_dataframe_format(self, trained_mock_model, synthetic_data):
        X, y = synthetic_data
        result = evaluate_features_shap(trained_mock_model, X, y)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col_name", "feature_importance_vals"]
        assert len(result) == 3
        top_feature = result.iloc[0]["col_name"]
        assert top_feature == "f1"

    def test_shap_3d_output(self, trained_mock_model, synthetic_data):
        X, y = synthetic_data
        # Mock shap.TreeExplainer to return 3D array
        with patch("shap.TreeExplainer") as mock_explainer:
            mock_inst = MagicMock()
            mock_inst.shap_values.return_value = np.random.randn(100, 3, 2)
            mock_explainer.return_value = mock_inst

            result = evaluate_features_shap(trained_mock_model, X, y)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3

    def test_shap_2d_output(self, trained_mock_model, synthetic_data):
        X, y = synthetic_data
        with patch("shap.TreeExplainer") as mock_explainer:
            mock_inst = MagicMock()
            mock_inst.shap_values.return_value = np.random.randn(100, 3)
            mock_explainer.return_value = mock_inst

            result = evaluate_features_shap(trained_mock_model, X, y)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3

    def test_shap_graceful_error_handling(self):
        class InvalidModel:
            pass

        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        result = evaluate_features_shap(InvalidModel(), X, y)
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestEvaluateFeaturesMDA:
    def test_mda_series_format(self, trained_mock_model, synthetic_data):
        X, y = synthetic_data
        result = evaluate_features_mda(trained_mock_model, X, y, n_repeats=3)
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert result.idxmax() == "f1"
        assert result["f1"] > 0.1

    def test_mda_graceful_error_handling(self):
        X = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
        y = pd.Series([0, 0])
        mock = DummyMetaClassifierMock()
        mock.fit(X, y)
        result = evaluate_features_mda(mock, X, y, n_repeats=1)
        assert isinstance(result, pd.Series)
        assert result.empty

    def test_mda_negative_and_neutral_scores(self, synthetic_data):
        X, y = synthetic_data
        # Mock model that returns predict_proba such that shuffling improves or doesn't change AUC
        mock = MagicMock()
        mock.predict_proba.return_value = np.array([[0.5, 0.5]] * len(X))

        with patch("src.modeling.feature_evaluation.roc_auc_score") as mock_auc:
            # First call baseline, then per col calls
            mock_auc.side_effect = [0.5, 0.6, 0.5, 0.5]

            # X has 3 cols. So we need baseline + 3 calls (assuming n_repeats=1)
            # baseline = 0.5.
            # col1 perm = 0.6 -> score = 0.5 - 0.6 = -0.1 (negative)
            # col2 perm = 0.5 -> score = 0.0 (neutral)
            # col3 perm = 0.5 -> score = 0.0
            result = evaluate_features_mda(mock, X, y, n_repeats=1)
            assert len(result) == 3

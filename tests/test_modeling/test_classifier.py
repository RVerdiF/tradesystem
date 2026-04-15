import numpy as np
import pandas as pd
import pytest
from src.modeling.classifier import MetaClassifier
from unittest.mock import patch, MagicMock


@pytest.fixture
def dataset():
    np.random.seed(42)
    X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y


def test_meta_classifier_init():
    clf = MetaClassifier(n_estimators=10)
    assert clf.n_estimators == 10
    assert not clf.is_fitted_


def test_meta_classifier_fit(dataset):
    X, y = dataset
    clf = MetaClassifier(n_estimators=10, max_depth=3)
    clf.fit(X, y)
    assert clf.is_fitted_


def test_meta_classifier_predict(dataset):
    X, y = dataset
    clf = MetaClassifier(n_estimators=10, max_depth=3)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_meta_classifier_predict_proba(dataset):
    X, y = dataset
    clf = MetaClassifier(n_estimators=10, max_depth=3)
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_meta_classifier_not_fitted_raises():
    clf = MetaClassifier()
    X = pd.DataFrame({"f1": [1]})
    with pytest.raises(RuntimeError):
        clf.predict(X)

    with pytest.raises(RuntimeError):
        clf.predict_proba(X)

    with pytest.raises(RuntimeError):
        clf.feature_importances()


def test_meta_classifier_feature_importances(dataset):
    X, y = dataset
    clf = MetaClassifier(n_estimators=10, max_depth=3)
    clf.fit(X, y)

    imps = clf.feature_importances(feature_names=["f1", "f2"])
    assert len(imps) == 2
    assert "f1" in imps.index
    assert "f2" in imps.index


def test_meta_classifier_feature_importances_no_names(dataset):
    X, y = dataset
    clf = MetaClassifier(n_estimators=10, max_depth=3)
    clf.fit(X, y)

    imps = clf.feature_importances()
    assert len(imps) == 2


def test_meta_classifier_evaluate(dataset):
    X, y = dataset
    clf = MetaClassifier(n_estimators=10, max_depth=3)
    clf.fit(X, y)

    report = clf.evaluate(X, y)
    assert isinstance(report, dict)
    assert "accuracy" in report
    assert "auc" in report


def test_meta_classifier_evaluate_single_class(dataset):
    X, _ = dataset
    y = pd.Series(np.zeros(len(X)))  # only one class
    clf = MetaClassifier(n_estimators=10, max_depth=3)

    # Needs to handle the fact that ROC AUC fails with 1 class
    with patch("src.modeling.classifier.roc_auc_score", side_effect=ValueError):
        clf.fit(X, y)
        report = clf.evaluate(X, y)
        assert np.isnan(report["auc"])


def test_meta_classifier_xgboost_imbalance_treatment(dataset):
    """Verifica que o XGBoost ajusta scale_pos_weight dinamicamente no método fit()
    para lidar corretamente com datasets desbalanceados.
    """
    X, _ = dataset
    # Create imbalanced dataset (90 neg, 10 pos -> ratio 9.0)
    y = pd.Series(np.concatenate([np.zeros(90), np.ones(10)]))

    clf = MetaClassifier(n_estimators=10, max_depth=3, class_weight="balanced", use_xgboost=True)

    # We will patch fit to bypass it and patch predict_proba to return dummy array
    # to avoid the NotFittedError from xgboost.
    with patch.object(clf.model, "fit"):
        with patch.object(clf.model, "predict_proba", return_value=np.zeros((100, 2))):
            with patch.object(clf.model, "set_params") as mock_set_params:
                clf.fit(X, y)
                # scale_pos_weight DEVE ser ajustado dinamicamente para n_neg / n_pos
                mock_set_params.assert_called_once()
                args, kwargs = mock_set_params.call_args
                assert "scale_pos_weight" in kwargs
                assert np.isclose(kwargs["scale_pos_weight"], 9.0)


def test_meta_classifier_custom_scale_pos_weight():
    """Verifica que o scale_pos_weight pode ser definido custommente e é passado para o XGBoost.
    """
    clf = MetaClassifier(n_estimators=10, max_depth=3, scale_pos_weight=2.5, use_xgboost=True)
    assert clf.scale_pos_weight == 2.5
    # Verify it's passed to the XGBClassifier model
    assert clf.model.scale_pos_weight == 2.5


def test_meta_classifier_rf_fallback(dataset):
    X, y = dataset
    clf = MetaClassifier(n_estimators=10, use_xgboost=False)
    clf.fit(X, y)
    assert clf.is_fitted_
    assert "RandomForestClassifier" in str(type(clf.model))


def test_meta_classifier_nan_warning(dataset):
    X, y = dataset
    # insere nans
    X_nan = X.copy()
    X_nan.iloc[0, 0] = np.nan
    clf = MetaClassifier(n_estimators=10, max_depth=3)
    # se XGB usar e lidar com nan tudo bem, mas verifica q passou na condicao
    # sem quebrar
    clf.fit(X_nan, y)
    assert clf.is_fitted_

import numpy as np
import pandas as pd
import pytest
from src.modeling.cv import PurgedKFold, cv_score
from unittest.mock import patch, MagicMock

def test_purged_k_fold_init():
    samples_info = pd.Series(
        pd.date_range("2024-01-02", periods=10, freq="1d"),
        index=pd.date_range("2024-01-01", periods=10, freq="1d")
    )
    cv = PurgedKFold(samples_info, n_splits=5, pct_embargo=0.01)
    assert cv.n_splits == 5
    assert cv.pct_embargo == 0.01
    
def test_purged_k_fold_init_invalid():
    with pytest.raises(ValueError):
        PurgedKFold(np.array([1, 2, 3]))

@patch("src.modeling.cv.get_train_times")
@patch("src.modeling.cv.apply_embargo")
def test_purged_k_fold_split(mock_apply_embargo, mock_get_train_times):
    # Setup
    dates = pd.date_range("2024-01-01", periods=10, freq="1d")
    samples_info = pd.Series(
        pd.date_range("2024-01-02", periods=10, freq="1d"),
        index=dates
    )
    
    cv = PurgedKFold(samples_info, n_splits=2)
    
    # Mocking purge/embargo returns
    # Just returning the input train_info for simplicity of testing the indices extraction
    mock_get_train_times.side_effect = lambda t, _: t
    mock_apply_embargo.side_effect = lambda t, _, __: t
    
    X = pd.DataFrame(np.random.randn(10, 2), index=dates)
    
    splits = list(cv.split(X))
    
    assert len(splits) == 2
    # Fold 1: train [5..9], test [0..4]
    train1, test1 = splits[0]
    assert len(test1) == 5
    assert list(test1) == [0, 1, 2, 3, 4]
    assert len(train1) == 5
    assert list(train1) == [5, 6, 7, 8, 9]

    # Fold 2: train [0..4], test [5..9]
    train2, test2 = splits[1]
    assert len(test2) == 5
    assert list(test2) == [5, 6, 7, 8, 9]
    assert len(train2) == 5
    assert list(train2) == [0, 1, 2, 3, 4]

def test_purged_k_fold_split_invalid_length():
    samples_info = pd.Series(
        pd.date_range("2024-01-02", periods=5, freq="1d"),
        index=pd.date_range("2024-01-01", periods=5, freq="1d")
    )
    cv = PurgedKFold(samples_info, n_splits=2)
    X = pd.DataFrame(np.random.randn(10, 2))
    with pytest.raises(ValueError):
        list(cv.split(X))

def test_purged_k_fold_split_ndarray():
    samples_info = pd.Series(
        pd.date_range("2024-01-02", periods=4, freq="1d"),
        index=pd.date_range("2024-01-01", periods=4, freq="1d")
    )
    cv = PurgedKFold(samples_info, n_splits=2)
    X = np.random.randn(4, 2)
    with patch("src.modeling.cv.get_train_times") as mock_get_train_times:
        with patch("src.modeling.cv.apply_embargo") as mock_apply_embargo:
            mock_get_train_times.side_effect = lambda t, _: t
            mock_apply_embargo.side_effect = lambda t, _, __: t
            splits = list(cv.split(X))
            assert len(splits) == 2

def test_cv_score_no_samples_info():
    clf = MagicMock()
    X = pd.DataFrame(np.random.randn(10, 2))
    y = pd.Series(np.random.randint(0, 2, 10))
    with pytest.raises(ValueError):
        cv_score(clf, X, y, None, samples_info=None)


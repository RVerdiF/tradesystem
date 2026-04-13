import numpy as np
import pandas as pd
import pytest
from src.labeling.meta_labeling import get_meta_labels, build_training_dataset, meta_label_analysis

@pytest.fixture
def tb_result():
    dates = pd.date_range("2024-01-01", periods=10, freq="1d")
    return pd.DataFrame({
        "ret": [0.05, -0.02, 0.01, -0.05, 0.03, 0.00, 0.06, -0.01, 0.04, -0.03],
        "label": [1, -1, 0, -1, 1, 0, 1, -1, 1, -1],
        "side": [1, 1, 1, -1, -1, -1, 1, 1, -1, -1],
        "barrier_type": ["pt", "sl", "vertical", "sl", "pt", "vertical", "pt", "sl", "pt", "sl"]
    }, index=dates)

@pytest.fixture
def tb_result_no_label():
    dates = pd.date_range("2024-01-01", periods=5, freq="1d")
    return pd.DataFrame({
        "ret": [0.05, -0.02, 0.01, -0.05, 0.03]
    }, index=dates)

@pytest.fixture
def features(tb_result):
    return pd.DataFrame({
        "f1": np.random.randn(len(tb_result)),
        "f2": np.random.randn(len(tb_result))
    }, index=tb_result.index)

def test_get_meta_labels(tb_result):
    meta = get_meta_labels(tb_result)
    assert not meta.empty
    assert set(meta.unique()).issubset({0, 1})
    assert meta.name == "meta_label"
    assert (meta == 1).sum() == 4 # 4 'pt' barriers have label=1

def test_get_meta_labels_no_label_col(tb_result_no_label):
    meta = get_meta_labels(tb_result_no_label, min_return=0.0)
    assert not meta.empty
    assert set(meta.unique()).issubset({0, 1})
    assert (meta == 1).sum() == 3 # > 0 returns

def test_build_training_dataset(features, tb_result):
    ds = build_training_dataset(features, tb_result)
    assert not ds.empty
    assert "meta_label" in ds.columns
    assert "side" in ds.columns
    assert "f1" in ds.columns
    
def test_build_training_dataset_return_info(features, tb_result):
    ds = build_training_dataset(features, tb_result, include_return_info=True)
    assert "_ret" in ds.columns
    assert "_barrier_type" in ds.columns

def test_meta_label_analysis(tb_result):
    stats = meta_label_analysis(tb_result)
    assert "total_events" in stats
    assert "positive_labels" in stats
    assert "barrier_breakdown" in stats
    assert "mean_return" in stats

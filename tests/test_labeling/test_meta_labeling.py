import numpy as np
import pandas as pd
import pytest

from src.labeling.meta_labeling import get_meta_labels


@pytest.fixture
def tb_result():
    dates = pd.date_range("2024-01-01", periods=10, freq="1d")
    return pd.DataFrame(
        {
            "ret": [0.05, -0.02, 0.01, -0.05, 0.03, 0.00, 0.06, -0.01, 0.04, -0.03],
            "label": [1, -1, 0, -1, 1, 0, 1, -1, 1, -1],
            "side": [1, 1, 1, -1, -1, -1, 1, 1, -1, -1],
            "barrier_type": [
                "pt",
                "sl",
                "vertical",
                "sl",
                "pt",
                "vertical",
                "pt",
                "sl",
                "pt",
                "sl",
            ],
        },
        index=dates,
    )


@pytest.fixture
def tb_result_no_label():
    dates = pd.date_range("2024-01-01", periods=5, freq="1d")
    return pd.DataFrame({"ret": [0.05, -0.02, 0.01, -0.05, 0.03]}, index=dates)


@pytest.fixture
def features(tb_result):
    return pd.DataFrame(
        {"f1": np.random.randn(len(tb_result)), "f2": np.random.randn(len(tb_result))},
        index=tb_result.index,
    )


def test_get_meta_labels(tb_result):
    meta = get_meta_labels(tb_result)
    assert not meta.empty
    assert set(meta.unique()).issubset({0, 1})
    assert meta.name == "meta_label"
    assert (meta == 1).sum() == 4  # 4 'pt' barriers have label=1


def test_get_meta_labels_no_label_col(tb_result_no_label):
    meta = get_meta_labels(tb_result_no_label, min_return=0.0)
    assert not meta.empty
    assert set(meta.unique()).issubset({0, 1})
    assert (meta == 1).sum() == 3  # > 0 returns

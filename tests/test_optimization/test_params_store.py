import json
import pytest
from pathlib import Path
from unittest.mock import patch
from src.optimization.params_store import save_optimized_params, load_optimized_params, params_exist

@pytest.fixture
def temp_models_dir(tmp_path):
    """Fixture to mock MODELS_DIR with a temporary directory."""
    test_dir = tmp_path / "models"
    test_dir.mkdir()
    with patch("src.optimization.params_store.MODELS_DIR", test_dir):
        yield test_dir

def test_save_and_load_params(temp_models_dir):
    """Test round-trip save and load of optimized parameters."""
    symbol = "PETR4"
    params = {"fast_span": 10, "slow_span": 50, "cusum_threshold": 0.02}
    metadata = {"best_sharpe": 1.5, "n_trials": 100}

    # Save
    path = save_optimized_params(symbol, params, metadata)
    assert path.exists()
    assert path.name == "params_PETR4.json"

    # Load
    loaded_data = load_optimized_params(symbol)
    assert loaded_data is not None
    assert loaded_data["symbol"] == symbol
    assert loaded_data["params"] == params
    assert loaded_data["metadata"]["best_sharpe"] == 1.5
    assert "timestamp" in loaded_data["metadata"]

def test_params_exist(temp_models_dir):
    """Test checking if parameters file exists."""
    symbol = "VALE3"
    assert not params_exist(symbol)

    save_optimized_params(symbol, {"param": 1})
    assert params_exist(symbol)
    # Check that it handles .SA suffix correctly
    assert params_exist("VALE3.SA")

def test_load_nonexistent_params(temp_models_dir):
    """Test loading parameters for a symbol that hasn't been saved."""
    assert load_optimized_params("NONEXISTENT") is None

def test_save_overwrites_existing(temp_models_dir):
    """Test that saving again overwrites the previous parameters."""
    symbol = "PETR4"
    save_optimized_params(symbol, {"v": 1})
    save_optimized_params(symbol, {"v": 2})

    loaded = load_optimized_params(symbol)
    assert loaded["params"]["v"] == 2

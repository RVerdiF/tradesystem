"""
Testes para src.optimization.params_store (SQLite backend).

Usa pytest tmp_path para isolar cada teste com um banco temporário,
evitando poluição do banco de desenvolvimento.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from src.optimization.params_store import (
    save_optimized_params,
    load_optimized_params,
    params_exist,
)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """
    Redireciona DB_PATH para um banco temporário por teste.
    Garante isolamento total entre testes sem atingir o banco real.
    """
    db_file = tmp_path / "test_params.db"
    # Patch em ambos os módulos que importam DB_PATH
    with (
        patch("src.db.DB_PATH", db_file),
        patch("src.optimization.params_store.get_connection") as mock_get_conn,
        patch("src.optimization.params_store.init_db") as mock_init,
    ):
        # Reimplementa get_connection usando o db temporário
        import sqlite3
        from src.db import _ALL_DDL

        def _temp_conn():
            conn = sqlite3.connect(str(db_file))
            conn.row_factory = sqlite3.Row
            for ddl in _ALL_DDL:
                conn.execute(ddl)
            conn.commit()
            return conn

        mock_get_conn.side_effect = _temp_conn
        mock_init.return_value = None
        yield db_file


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip():
    """Save + Load deve recuperar os dados exatos."""
    symbol = "PETR4"
    params = {"fast_span": 10, "slow_span": 50, "cusum_threshold": 0.02}
    metadata = {"best_sharpe": 1.5, "n_trials": 100}

    save_optimized_params(symbol, params, metadata)
    loaded = load_optimized_params(symbol)

    assert loaded is not None
    assert loaded["symbol"] == symbol
    assert loaded["params"] == params
    assert loaded["metadata"]["best_sharpe"] == 1.5
    assert "timestamp" in loaded["metadata"]


def test_params_exist_true_after_save():
    assert not params_exist("VALE3")
    save_optimized_params("VALE3", {"p": 1})
    assert params_exist("VALE3")


def test_sa_suffix_normalization():
    """VALE3.SA e VALE3 devem referenciar o mesmo registro."""
    save_optimized_params("VALE3.SA", {"param": 42})
    assert params_exist("VALE3")
    assert params_exist("VALE3.SA")
    data = load_optimized_params("VALE3")
    assert data["symbol"] == "VALE3"


def test_load_nonexistent_returns_none():
    assert load_optimized_params("NONEXISTENT") is None


def test_save_overwrites_existing():
    """Segunda gravação deve substituir a primeira (INSERT OR REPLACE)."""
    symbol = "PETR4"
    save_optimized_params(symbol, {"v": 1})
    save_optimized_params(symbol, {"v": 2})

    loaded = load_optimized_params(symbol)
    assert loaded["params"]["v"] == 2


def test_metadata_timestamp_auto_added():
    """timestamp deve ser adicionado automaticamente pelo save."""
    save_optimized_params("WINFUT", {"x": 0})
    data = load_optimized_params("WINFUT")
    assert "timestamp" in data["metadata"]


def test_save_without_metadata():
    """Salvar sem metadata explícito deve funcionar."""
    save_optimized_params("WDOFUT", {"y": 5}, metadata=None)
    data = load_optimized_params("WDOFUT")
    assert data is not None
    assert "timestamp" in data["metadata"]

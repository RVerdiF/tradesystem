"""Testes para o módulo de armazenamento Parquet (storage.py).

Testa round-trip save/load, append sem duplicatas, filtro temporal e metadados.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.storage import ParquetStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def store(tmp_path):
    """ParquetStore apontando para diretório temporário."""
    return ParquetStore(base_dir=tmp_path)


@pytest.fixture
def sample_ohlc():
    """DataFrame OHLC de exemplo com 10 barras."""
    dates = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100 + i for i in range(10)],
            "high": [102 + i for i in range(10)],
            "low": [99 + i for i in range(10)],
            "close": [101 + i for i in range(10)],
            "tick_volume": [1000 + i * 10 for i in range(10)],
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------
class TestParquetStoreRoundTrip:
    """Testes de save/load (round-trip)."""

    def test_save_and_load_preserves_data(self, store, sample_ohlc):
        """Salvar e recarregar deve preservar todos os dados."""
        store.save(sample_ohlc, "PETR4", "ohlc", append=False)

        loaded = store.load("PETR4", "ohlc")

        pd.testing.assert_frame_equal(loaded, sample_ohlc, check_freq=False)

    def test_save_creates_file(self, store, sample_ohlc):
        """Após salvar, o arquivo deve existir."""
        assert not store.exists("PETR4", "ohlc")

        store.save(sample_ohlc, "PETR4", "ohlc")

        assert store.exists("PETR4", "ohlc")

    def test_load_nonexistent_raises(self, store):
        """Carregar dados inexistentes deve levantar FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            store.load("INEXISTENTE", "ohlc")


class TestParquetStoreAppend:
    """Testes de append."""

    def test_append_adds_new_rows(self, store, sample_ohlc):
        """Append deve adicionar registros novos sem duplicar existentes."""
        # Salva primeiras 5 barras
        store.save(sample_ohlc.iloc[:5], "VALE3", "ohlc", append=False)

        # Append das últimas 7 barras (2 sobrepostas)
        store.save(sample_ohlc.iloc[3:], "VALE3", "ohlc", append=True)

        loaded = store.load("VALE3", "ohlc")

        # Deve ter exatamente 10 barras (sem duplicatas)
        assert len(loaded) == 10

    def test_append_preserves_order(self, store, sample_ohlc):
        """Dados após append devem estar ordenados por tempo."""
        store.save(sample_ohlc.iloc[5:], "WIN", "ohlc", append=False)
        store.save(sample_ohlc.iloc[:5], "WIN", "ohlc", append=True)

        loaded = store.load("WIN", "ohlc")

        assert loaded.index.is_monotonic_increasing


class TestParquetStoreFilter:
    """Testes de filtro temporal."""

    def test_load_with_start_filter(self, store, sample_ohlc):
        """Filtro start deve retornar apenas registros posteriores."""
        store.save(sample_ohlc, "PETR4", "ohlc")

        mid_time = sample_ohlc.index[5]
        loaded = store.load("PETR4", "ohlc", start=mid_time)

        assert loaded.index.min() >= mid_time

    def test_load_with_end_filter(self, store, sample_ohlc):
        """Filtro end deve retornar apenas registros anteriores."""
        store.save(sample_ohlc, "PETR4", "ohlc")

        mid_time = sample_ohlc.index[5]
        loaded = store.load("PETR4", "ohlc", end=mid_time)

        assert loaded.index.max() <= mid_time

    def test_load_with_start_and_end(self, store, sample_ohlc):
        """Filtro combinado start+end deve restringir janela."""
        store.save(sample_ohlc, "PETR4", "ohlc")

        start = sample_ohlc.index[2]
        end = sample_ohlc.index[7]
        loaded = store.load("PETR4", "ohlc", start=start, end=end)

        assert loaded.index.min() >= start
        assert loaded.index.max() <= end


class TestParquetStoreUtils:
    """Testes de utilitários."""

    def test_list_symbols(self, store, sample_ohlc):
        """list_symbols deve retornar todos os símbolos salvos."""
        store.save(sample_ohlc, "PETR4", "ohlc")
        store.save(sample_ohlc, "VALE3", "ohlc")

        symbols = store.list_symbols("ohlc")

        assert set(symbols) == {"PETR4", "VALE3"}

    def test_delete_removes_data(self, store, sample_ohlc):
        """Delete deve remover o arquivo."""
        store.save(sample_ohlc, "PETR4", "ohlc")
        assert store.exists("PETR4", "ohlc")

        store.delete("PETR4", "ohlc")
        assert not store.exists("PETR4", "ohlc")

    def test_info_returns_metadata(self, store, sample_ohlc):
        """Info deve retornar metadados do arquivo."""
        store.save(sample_ohlc, "PETR4", "ohlc")

        info = store.info("PETR4", "ohlc")

        assert info["exists"] is True
        assert info["num_rows"] == 10
        assert info["size_mb"] > 0

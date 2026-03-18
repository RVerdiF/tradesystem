"""
1.3 — Armazenamento em Parquet.

Classe ``ParquetStore`` para leitura e escrita de alta performance em Parquet,
com particionamento por símbolo e data, compressão snappy e append eficiente.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR


class ParquetStore:
    """
    Gerencia armazenamento de DataFrames em formato Parquet.

    Organização no disco::

        base_dir/
        └── {data_type}/
            └── {symbol}/
                └── {symbol}_{data_type}.parquet

    Parameters
    ----------
    base_dir : Path, optional
        Diretório raiz para armazenamento. Default: ``data/processed``.
    compression : str
        Algoritmo de compressão (``snappy``, ``gzip``, ``zstd``). Default: ``snappy``.
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        compression: str = "snappy",
    ) -> None:
        self.base_dir = base_dir or PROCESSED_DATA_DIR
        self.compression = compression

    # ---- Escrita ----

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_type: str = "ohlc",
        append: bool = True,
    ) -> Path:
        """
        Salva um DataFrame em Parquet.

        Parameters
        ----------
        df : pd.DataFrame
            Dados a salvar (index deve ser DatetimeIndex).
        symbol : str
            Símbolo do ativo.
        data_type : str
            Tipo de dados (``ohlc``, ``ticks``, ``volume_bars``, etc.).
        append : bool
            Se True, concatena com dados existentes (sem duplicatas).

        Returns
        -------
        Path
            Caminho do arquivo salvo.
        """
        file_path = self._resolve_path(symbol, data_type)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if append and file_path.exists():
            existing_df = self.load(symbol, data_type)
            df = pd.concat([existing_df, df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)

        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path, compression=self.compression)

        size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Salvo: {} | {} | {} registros | {:.2f} MB",
            symbol,
            data_type,
            len(df),
            size_mb,
        )
        return file_path

    # ---- Leitura ----

    def load(
        self,
        symbol: str,
        data_type: str = "ohlc",
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Carrega dados de um arquivo Parquet.

        Parameters
        ----------
        symbol : str
            Símbolo do ativo.
        data_type : str
            Tipo de dados.
        start : str or Timestamp, optional
            Filtro de data inicial (inclusive).
        end : str or Timestamp, optional
            Filtro de data final (inclusive).
        columns : list[str], optional
            Subconjunto de colunas a carregar.

        Returns
        -------
        pd.DataFrame
            DataFrame com os dados carregados.

        Raises
        ------
        FileNotFoundError
            Se o arquivo não existe.
        """
        file_path = self._resolve_path(symbol, data_type)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dados não encontrados: {file_path}. "
                f"Execute a extração primeiro."
            )

        table = pq.read_table(file_path, columns=columns)
        df = table.to_pandas()

        # Restaura DatetimeIndex se necessário
        if "time" in df.columns:
            df.set_index("time", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            # Tenta converter o index
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass

        # Filtro temporal
        if start is not None:
            df = df.loc[start:]  # type: ignore[misc]
        if end is not None:
            df = df.loc[:end]  # type: ignore[misc]

        logger.debug(
            "Carregado: {} | {} | {} registros",
            symbol,
            data_type,
            len(df),
        )
        return df

    # ---- Utilitários ----

    def exists(self, symbol: str, data_type: str = "ohlc") -> bool:
        """Verifica se dados existem para o símbolo/tipo."""
        return self._resolve_path(symbol, data_type).exists()

    def list_symbols(self, data_type: str = "ohlc") -> list[str]:
        """Lista todos os símbolos disponíveis para um tipo de dados."""
        type_dir = self.base_dir / data_type
        if not type_dir.exists():
            return []
        return [d.name for d in type_dir.iterdir() if d.is_dir()]

    def delete(self, symbol: str, data_type: str = "ohlc") -> bool:
        """Remove dados de um símbolo/tipo."""
        file_path = self._resolve_path(symbol, data_type)
        if file_path.exists():
            file_path.unlink()
            logger.info("Removido: {} | {}", symbol, data_type)
            return True
        return False

    def info(self, symbol: str, data_type: str = "ohlc") -> dict:
        """Retorna metadados sobre o arquivo Parquet."""
        file_path = self._resolve_path(symbol, data_type)
        if not file_path.exists():
            return {"exists": False}

        metadata = pq.read_metadata(file_path)
        return {
            "exists": True,
            "path": str(file_path),
            "size_mb": file_path.stat().st_size / (1024 * 1024),
            "num_rows": metadata.num_rows,
            "num_columns": metadata.num_columns,
            "num_row_groups": metadata.num_row_groups,
            "created_by": metadata.created_by,
        }

    # ---- Interno ----

    def _resolve_path(self, symbol: str, data_type: str) -> Path:
        """Monta o caminho do arquivo Parquet."""
        return self.base_dir / data_type / symbol / f"{symbol}_{data_type}.parquet"

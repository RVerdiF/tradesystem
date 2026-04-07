"""
Gerenciamento e persistência de hiperparâmetros otimizados.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from loguru import logger

# Configura o diretório padrão para "models" no root do projeto
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _get_params_path(symbol: str) -> Path:
    """Retorna o caminho completo do arquivo de parâmetros para o símbolo."""
    # Garante a remoção do .SA para ficar coerente com o padrão MT5 adotado
    symbol = symbol.replace(".SA", "")
    return MODELS_DIR / f"params_{symbol}.json"


def save_optimized_params(symbol: str, params: dict, metadata: dict | None = None) -> Path:
    """
    Salva os parâmetros otimizados e metadados num arquivo JSON.
    
    Parameters
    ----------
    symbol : str
        Ativo correspondente (ex: PETR4)
    params : dict
        Hiperparâmetros otimizados
    metadata : dict | None
        Outros dados do Optuna (Sharpe, DSR, data, n_trials)
    """
    path = _get_params_path(symbol)
    
    if metadata is None:
        metadata = {}
        
    # Adicionar timestamp
    metadata["timestamp"] = datetime.now().isoformat()
    
    data = {
        "symbol": symbol,
        "metadata": metadata,
        "params": params
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    logger.success(f"Parâmetros otimizados de {symbol} salvos em {path}")
    return path


def load_optimized_params(symbol: str) -> dict | None:
    """
    Lê os parâmetros de um arquivo JSON.
    Retorna o dicionário com 'params' e 'metadata', ou None se não existir.
    """
    path = _get_params_path(symbol)
    
    if not path.exists():
        logger.info(f"Nenhum arquivo de parâmetros encontrado para {symbol}")
        return None
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Parâmetros de {symbol} carregados (Otimizados em: {data.get('metadata', {}).get('timestamp', 'N/A')})")
        return data
    except Exception as e:
        logger.error(f"Erro ao carregar parâmetros de {symbol} em {path}: {e}")
        return None


def params_exist(symbol: str) -> bool:
    """Verifica se existe um arquivo de parâmetros para o símbolo."""
    return _get_params_path(symbol).exists()

"""
Configurações centralizadas do TradeSystem5000.

Credenciais sensíveis são lidas de variáveis de ambiente.
Parâmetros de trading e risco são definidos aqui como constantes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Diretórios
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DATA_DIR = DATA_DIR / "samples"
LOG_DIR = PROJECT_ROOT / "logs"

# Garante que diretórios existam
for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLES_DATA_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# MT5
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MT5Config:
    """Configuração de conexão com o MetaTrader 5."""

    login: int = int(os.getenv("MT5_LOGIN", "0"))
    password: str = os.getenv("MT5_PASSWORD", "")
    server: str = os.getenv("MT5_SERVER", "")
    path: str = os.getenv("MT5_PATH", "")  # caminho do terminal64.exe
    timeout: int = int(os.getenv("MT5_TIMEOUT", "10000"))
    max_retries: int = 3
    retry_delay: float = 2.0  # segundos entre tentativas


# ---------------------------------------------------------------------------
# Símbolos e Timeframes
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS: list[str] = [
    "PETR4",
    "VALE3",
    "WINFUT",
    "WDOFUT",
]

# Timeframes MT5 (constantes numéricas do MetaTrader5)
# 1=M1, 5=M5, 15=M15, 60=H1, 1440=D1
DEFAULT_TIMEFRAME: int = 5  # M5


# ---------------------------------------------------------------------------
# Barras alternativas
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BarSamplingConfig:
    """Parâmetros para amostragem de barras de volume e dólar."""

    volume_bar_threshold: int = 1000       # ticks acumulados para barra de volume
    dollar_bar_threshold: float = 1_000_000.0  # valor financeiro para barra de dólar


# ---------------------------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CleaningConfig:
    """Parâmetros de limpeza de dados."""

    spike_z_threshold: float = 5.0   # Z-score para considerar spike
    max_gap_seconds: int = 300       # lacuna máxima tolerada (5 min)
    fill_method: str = "ffill"       # forward fill


# ---------------------------------------------------------------------------
# Risco
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskConfig:
    """Parâmetros de gerenciamento de risco e circuit breakers."""

    max_daily_loss_pct: float = 0.02       # 2% do capital
    max_drawdown_pct: float = 0.05         # 5% drawdown máximo
    max_position_size: float = 1.0         # lote máximo por trade
    max_open_positions: int = 5
    kelly_fraction: float = 0.5            # Kelly fracionário (50%)


# ---------------------------------------------------------------------------
# Custos (mercado brasileiro — B3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CostConfig:
    """Taxas operacionais para modelagem de custos."""

    brokerage_per_contract: float = 0.0    # corretagem (muitas corretoras = zero)
    emoluments_pct: float = 0.00005        # emolumentos B3
    settlement_pct: float = 0.0000275      # liquidação
    iss_pct: float = 0.05                  # ISS sobre corretagem
    slippage_bps: float = 1.0              # slippage estimado em basis points


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LogConfig:
    """Configuração de logging."""

    level: str = "INFO"
    rotation: str = "10 MB"
    retention: str = "30 days"
    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )


# ---------------------------------------------------------------------------
# Instâncias padrão (singleton-like)
# ---------------------------------------------------------------------------
mt5_config = MT5Config()
bar_sampling_config = BarSamplingConfig()
cleaning_config = CleaningConfig()
risk_config = RiskConfig()
cost_config = CostConfig()
log_config = LogConfig()

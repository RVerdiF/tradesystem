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
DB_PATH = PROJECT_ROOT / "data" / "tradesystem.db"

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
# Feature Engineering
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    """Parâmetros de Feature Engineering (Fase 2)."""

    # Diferenciação Fracionária (FFD)
    ffd_d: float = 0.4              # d inicial para FFD
    ffd_threshold: float = 1e-4     # corte de pesos (Ajustado p/ 1 ano)
    ffd_adf_pvalue: float = 0.05    # p-value alvo do teste ADF

    # CUSUM
    cusum_threshold_pct: float = 0.0435  # % de desvio para trigger (Otimizado PETR4.SA)
    cusum_ewm_span: int = 50          # span do EWMA para threshold adaptativo

    # Indicadores — Momentum
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Indicadores — Volatilidade
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0

    # Normalização
    zscore_window: int = 50

# ---------------------------------------------------------------------------
# Labeling (Fase 3 — Alpha + Tripla Barreira)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingConfig:
    """Parâmetros para rotulagem e alpha models (Fase 3)."""

    # Alpha — Trend Following
    trend_fast_span: int = 11        # EMA rápida (períodos) (Otimizado PETR4.SA)
    trend_slow_span: int = 58        # EMA lenta (períodos) (Otimizado PETR4.SA)

    # Alpha — Mean Reversion
    mean_rev_window: int = 20        # Janela do Z-score
    mean_rev_entry: float = 2.0      # Z-score de entrada (|z| > entry → sinal)
    mean_rev_exit: float = 0.0       # Z-score de saída

    # Volatilidade dinâmica
    vol_span: int = 20               # Span EWMA para volatilidade dos retornos

    # Tripla Barreira
    pt_sl_ratio: tuple[float, float] = (2.77, 2.98)  # (profit_take, stop_loss) (Otimizado PETR4.SA)
    max_holding_periods: int = 10    # Barreira vertical (barras máximas)
    min_return: float = 0.0          # Retorno mínimo para considerar label +1


# ---------------------------------------------------------------------------
# ML Modeling (Fase 4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MLConfig:
    """Parâmetros para modelagem ML e Validação Cruzada (Fase 4)."""

    # Validação Cruzada
    cv_splits: int = 5
    embargo_pct: float = 0.01        # 1% das barras da base como embargo pós-teste
    xgb_max_depth: int = 4           # Profundidade máxima XGBoost (Otimizado PETR4.SA)

    # Bet Sizing
    max_leverage: int = 5            # Alavancagem máxima (lotes)


# ---------------------------------------------------------------------------
# Otimização (Fase Extra)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptimizationConfig:
    """Configuração para o otimizador bayesiano (Optuna)."""

    # Ranges de busca sugeridos (Fase 1 do plano)
    cusum_range: tuple[float, float] = (0.01, 0.05)
    fast_span_range: tuple[int, int] = (5, 20)
    slow_span_range: tuple[int, int] = (20, 60)
    pt_sl_range: tuple[float, float] = (1.0, 3.0)
    max_depth_range: tuple[int, int] = (2, 4)

    # Registros adicionais (Features)
    rsi_period_range: tuple[int, int] = (7, 28)
    macd_fast_range: tuple[int, int] = (6, 20)
    macd_slow_range: tuple[int, int] = (20, 40)
    macd_signal_range: tuple[int, int] = (5, 15)
    atr_period_range: tuple[int, int] = (7, 28)
    bb_period_range: tuple[int, int] = (10, 40)
    bb_std_range: tuple[float, float] = (1.5, 3.0)
    zscore_window_range: tuple[int, int] = (20, 100)
    ffd_d_range: tuple[float, float] = (0.1, 0.7)

    # Parâmetros de execução
    n_trials: int = 80
    min_trades: int = 30
    timeout: int = 5400  # 1.5 horas


# ---------------------------------------------------------------------------
# Execução (Fase 6 — Paper Trading / Live)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutionConfig:
    """Parâmetros do motor de execução."""

    mode: str = "paper"              # 'paper' ou 'live'
    poll_interval: float = 0.5       # Segundos entre leitura de ticks
    max_slippage_ticks: int = 5      # Desvio máximo aceito em envio a mercado
    magic_number: int = 5000         # Identificador das ordens do sistema
    reconciliation_interval: int = 60# Segundos entre reconciliações posição real vs esperada


# ---------------------------------------------------------------------------
# Instâncias padrão (singleton-like)
# ---------------------------------------------------------------------------
mt5_config = MT5Config()
bar_sampling_config = BarSamplingConfig()
cleaning_config = CleaningConfig()
risk_config = RiskConfig()
cost_config = CostConfig()
log_config = LogConfig()
feature_config = FeatureConfig()
labeling_config = LabelingConfig()
ml_config = MLConfig()
optimization_config = OptimizationConfig()
execution_config = ExecutionConfig()



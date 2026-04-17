"""Configurações centralizadas do TradeSystem5000.

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
DB_PATH = PROJECT_ROOT / "data" / "tradesystem.db"

# Garante que diretórios existam
for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR):
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
# Timeframes MT5 (constantes numéricas do MetaTrader5)
# 1=M1, 5=M5, 15=M15, 60=H1, 1440=D1
DEFAULT_TIMEFRAME: int = 5  # M5


# ---------------------------------------------------------------------------
# Barras alternativas
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BarSamplingConfig:
    """Parâmetros para amostragem de barras de volume e dólar."""

    volume_bar_threshold: int = 1000  # ticks acumulados para barra de volume
    dollar_bar_threshold: float = 1_000_000.0  # valor financeiro para barra de dólar


# ---------------------------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CleaningConfig:
    """Parâmetros de limpeza de dados."""

    spike_z_threshold: float = 5.0  # Z-score para considerar spike
    max_gap_seconds: int = 300  # lacuna máxima tolerada (5 min)
    fill_method: str = "ffill"  # forward fill


# ---------------------------------------------------------------------------
# Risco
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskConfig:
    """Parâmetros de gerenciamento de risco e circuit breakers."""

    max_daily_loss_pct: float = 0.02  # 2% do capital
    max_drawdown_pct: float = 0.05  # 5% drawdown máximo
    max_daily_profit_pct: float = 0.02  # 2% lucro máximo diário (novo)
    cool_down_minutes: float = 5.0  # Minutos de resfriamento pós-saída (novo)
    kelly_fraction: float = 0.5  # Kelly fracionário (50%)
    min_conviction_threshold: float = (
        0.65  # Limiar mínimo de probabilidade do Meta-Model para operar
    )

    # Horários e Modalidade
    trading_start_time: str = "09:00:00"
    trading_end_time: str = "17:55:00"
    trade_type: str = "day_trade"  # 'day_trade' ou 'swing_trade'


# ---------------------------------------------------------------------------
# Custos (mercado brasileiro — B3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CostConfig:
    """Taxas operacionais para modelagem de custos."""

    brokerage_per_contract: float = 0.0  # corretagem por contrato/ação
    emoluments_fixed: float = 0.25  # emolumentos fixos B3 (por minicontrato WIN)
    emoluments_pct: float = 0.00005  # emolumentos B3 (ações)
    settlement_pct: float = 0.0000275  # liquidação (ações)
    iss_pct: float = 0.05  # ISS sobre corretagem
    slippage_ticks: float = 1.0  # ticks de slippage por ordem para futuros
    slippage_bps: float = 2.0  # base slippage para ações

    # Configurações de ativos
    tick_sizes: dict[str, float] = field(default_factory=lambda: {"WIN": 5.0, "WDO": 0.5})
    asset_multipliers: dict[str, float] = field(default_factory=lambda: {"WIN": 0.2, "WDO": 10.0})


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    """Parâmetros de Feature Engineering (Fase 2)."""

    # Diferenciação Fracionária (FFD)
    ffd_d: float = 0.4  # d inicial para FFD
    ffd_threshold: float = 1e-4  # corte de pesos (Ajustado p/ 1 ano)
    ffd_adf_pvalue: float = 0.05  # p-value alvo do teste ADF

    # CUSUM
    cusum_threshold_pct: float = 0.0435  # % de desvio para trigger (Otimizado PETR4.SA)
    cusum_ewm_span: int = 50  # span do EWMA para threshold adaptativo

    # Indicadores — Momentum (Fase True Test)
    ma_dist_fast_period: int = 9
    ma_dist_slow_period: int = 21

    # Indicadores — Volatilidade e Estatística
    atr_period: int = 14
    moments_window: int = 40

    # Normalização
    zscore_window: int = 50

    # Regime Detection (Hurst Exponent)
    hurst_window: int = 100  # Janela rolante para cálculo do Hurst
    hurst_step: int = 5  # Passo de cálculo (>1 para performance)
    long_hurst_threshold: float = 0.55
    short_hurst_threshold: float = 0.55

    # Volume Imbalance (Order Flow Filter)
    long_vol_imbalance_z_threshold: float = 1.0
    short_vol_imbalance_z_threshold: float = 1.0

    # VPIN
    vpin_bucket_size: int = 5000
    vpin_window: int = 50


# ---------------------------------------------------------------------------
# Labeling (Fase 3 — Alpha + Tripla Barreira)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingConfig:
    """Parâmetros para rotulagem e alpha models (Fase 3)."""

    # Alpha — Trend Following
    trend_fast_span: int = 11  # EMA rápida (períodos) (Otimizado PETR4.SA)
    trend_slow_span: int = 58  # EMA lenta (períodos) (Otimizado PETR4.SA)

    # Alpha — Composite Alpha (Decoupled Long/Short)
    long_fast_span: int = 11
    long_slow_span: int = 58
    short_fast_span: int = 11
    short_slow_span: int = 58

    # Alpha — Mean Reversion
    mean_rev_window: int = 20  # Janela do Z-score
    mean_rev_entry: float = 2.0  # Z-score de entrada (|z| > entry → sinal)
    mean_rev_exit: float = 0.0  # Z-score de saída

    # Volatilidade dinâmica
    vol_span: int = 20  # Span EWMA para volatilidade dos retornos

    # Tripla Barreira
    pt_sl_ratio: tuple[float, float] = (2.77, 2.98)  # (profit_take, stop_loss) (Otimizado PETR4.SA)
    max_holding_periods: int = 100  # Barreira vertical (barras máximas)
    min_return: float = 0.005  # Retorno mínimo para considerar label +1


# ---------------------------------------------------------------------------
# ML Modeling (Fase 4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MLConfig:
    """Parâmetros para modelagem ML e Validação Cruzada (Fase 4)."""

    # Validação Cruzada
    cv_splits: int = 5
    embargo_pct: float = 0.03  # 5% — excede max_holding_periods e rompe autocorrelação
    xgb_max_depth: int = 2  # Produção: raso para evitar memorização de padrões espúrios
    xgb_gamma: float = 0.0  # Desativado (evita árvores vazias em amostras pequenas)
    xgb_min_child_weight: float = (
        5.0  # Balanceado: impede splits espúrios sem bloquear aprendizado em datasets ~300 amostras
    )
    xgb_lambda: float = 1.0  # L2 Regularization
    xgb_alpha: float = 0.0  # L1 Regularization

    # Bet Sizing
    max_leverage: int = 100  # Alavancagem máxima (lotes)


# ---------------------------------------------------------------------------
# Otimização (Fase Extra)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptimizationConfig:
    """Configuração para o otimizador bayesiano (Optuna)."""

    # Ranges de busca fundamentais (Top 10 - "Faxina Real")
    cusum_range: tuple[float, float] = (0.001, 0.003)
    pt_sl_range: tuple[float, float] = (0.5, 3.5)  # SL floor=1.5 em tuner.py
    meta_threshold_range: tuple[float, float] = (
        0.05,
        0.45,
    )  # calibrado para datasets desbalanceados (5-10% positivos)
    max_depth_range: tuple[int, int] = (1, 2)

    # Composite Alpha (Decoupled Ranges)
    long_fast_span_range: tuple[int, int] = (3, 8)
    long_slow_span_range: tuple[int, int] = (15, 30)
    short_fast_span_range: tuple[int, int] = (3, 8)
    short_slow_span_range: tuple[int, int] = (15, 30)
    long_hurst_threshold_range: tuple[float, float] = (0.50, 0.70)
    short_hurst_threshold_range: tuple[float, float] = (0.50, 0.70)
    long_vir_threshold_range: tuple[float, float] = (0.2, 2.0)
    short_vir_threshold_range: tuple[float, float] = (0.2, 2.0)

    ma_dist_fast_range: tuple[int, int] = (7, 15)
    ma_dist_slow_range: tuple[int, int] = (20, 40)
    moments_window_range: tuple[int, int] = (20, 100)

    be_trigger_range: tuple[float, float] = (0.0, 0.5)
    xgb_gamma_range: tuple[float, float] = (0.0, 2.0)
    xgb_lambda_range: tuple[float, float] = (1.0, 5.0)
    xgb_alpha_range: tuple[float, float] = (0.0, 2.0)
    xgb_min_child_weight_range: tuple[float, float] = (1.0, 20.0)
    ffd_d_range: tuple[float, float] = (0.1, 0.9)
    atr_period_range: tuple[int, int] = (7, 21)

    # Parâmetros de execução
    n_trials_phase1: int = 150  # Trials para otimização de Alpha/Features
    n_trials_phase2: int = 50  # Trials para otimização do Meta-Model
    min_trades: int = 30
    timeout: int = 5400  # 1.5 horas

    # Meta-Model hyperparameters for Phase 2
    scale_pos_weight_range: tuple[float, float] = (
        1.0,
        25.0,
    )  # range amplo: cobre desde datasets balanceados até 25:1 de desbalanceamento


# ---------------------------------------------------------------------------
# Execução (Fase 6 — Paper Trading / Live)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutionConfig:
    """Parâmetros do motor de execução."""

    mode: str = "live"  # 'paper' ou 'live'
    poll_interval: float = 0.5  # Segundos entre leitura de ticks
    max_slippage_ticks: int = 5  # Desvio máximo aceito em envio a mercado
    magic_number: int = 5000  # Identificador das ordens do sistema
    live_bars: int = 1000  # Barras buscadas do MT5 por ciclo (deve exceder min_bars do pipeline)


# ---------------------------------------------------------------------------
# Instâncias padrão (singleton-like)
# ---------------------------------------------------------------------------
mt5_config = MT5Config()
bar_sampling_config = BarSamplingConfig()
cleaning_config = CleaningConfig()
risk_config = RiskConfig()
cost_config = CostConfig()
feature_config = FeatureConfig()
labeling_config = LabelingConfig()
ml_config = MLConfig()
optimization_config = OptimizationConfig()
execution_config = ExecutionConfig()

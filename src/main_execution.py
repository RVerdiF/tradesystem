"""
Entry point para execução em Tempo Real (Live / Paper Trading).

Fluxo:
  1. Treina o Meta-Modelo usando dados históricos (yfinance ou MT5).
  2. Serializa o modelo treinado em disco (model.pkl).
  3. Monta um pipeline callable que encapsula: Features → Alpha → Meta → Kelly.
  4. Inicia o AsyncTradingEngine com o pipeline.

Uso:
  # Treinar modelo e iniciar paper trading:
  python -m src.main_execution --symbol PETR4.SA --interval 1h

  # Reusar modelo já treinado:
  python -m src.main_execution --symbol PETR4.SA --interval 1h --load-model models/model_PETR4.SA.pkl
"""

from __future__ import annotations

import argparse
import asyncio
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Configurações
from config.settings import (
    execution_config,
    labeling_config,
    feature_config,
    ml_config,
)

# Pipeline de dados e features
from src.data.mt5_connector import mt5_session
from src.data.extractor import extract_ohlc, INTERVAL_TO_TF
from src.features.indicators import compute_all_features
from src.features.frac_diff import frac_diff_ffd, find_min_d
from src.features.cusum_filter import adaptive_cusum_events

# Alpha e Labeling
from src.labeling.alpha import TrendFollowingAlpha, get_signal_events
from src.labeling.volatility import get_volatility_targets
from src.labeling.triple_barrier import create_events, get_labels

# ML
from src.modeling.classifier import MetaClassifier
from src.modeling.bet_sizing import compute_kelly_fraction

# Optimization Store
from src.optimization.params_store import (
    save_optimized_params,
    load_optimized_params,
    params_exist,
)

# Execução
from src.execution.engine import AsyncTradingEngine


# ---------------------------------------------------------------------------
# 1. Treinamento do modelo
# ---------------------------------------------------------------------------
def train_model(df: pd.DataFrame, interval: str = "1h", params: dict | None = None) -> dict:
    """
    Treina o Meta-Modelo usando o pipeline completo e retorna os artefatos
    necessários para execução em tempo real.

    Returns
    -------
    dict com chaves:
        - "model": MetaClassifier treinado
        - "optimal_d": d ótimo para FFD
        - "alpha": instância do TrendFollowingAlpha
        - "feature_columns": lista de nomes das features usadas
    """
    logger.info("=== Treinamento do Meta-Modelo para Execução ===")

    # --- Features ---
    features = compute_all_features(df)
    optimal_d = find_min_d(df["close"])
    features["ffd"] = frac_diff_ffd(df["close"], d=optimal_d)
    features = features.dropna()
    df_aligned = df.reindex(features.index)
    logger.info(f"Features: {features.shape}")

    if params is None:
        params = {}
        
    fast_span = params.get("alpha_fast", labeling_config.trend_fast_span)
    slow_span = params.get("alpha_slow", labeling_config.trend_slow_span)

    # --- Alpha ---
    alpha = TrendFollowingAlpha(
        fast_span=fast_span,
        slow_span=slow_span,
    )
    signal = alpha.generate_signal(df_aligned)
    signal_events = get_signal_events(signal)

    # CUSUM filter para aumentar eventos
    cusum_threshold = params.get("cusum_threshold", feature_config.cusum_threshold_pct)
    cusum_ts = adaptive_cusum_events(
        df_aligned["close"],
        threshold_multiplier=cusum_threshold,
    )
    # União: Alpha events + CUSUM events (mais amostras para treinar)
    all_events = signal_events.union(cusum_ts).sort_values()
    logger.info(f"Eventos totais (Alpha ∪ CUSUM): {len(all_events)}")

    if len(all_events) == 0:
        logger.error("Nenhum evento gerado. Abortando treinamento.")
        sys.exit(1)

    target_vol = get_volatility_targets(
        df_aligned["close"], all_events, span=labeling_config.vol_span
    )

    # Direção do alpha nos eventos (ffill para CUSUM events sem cruzamento)
    side = signal.reindex(target_vol.index, method="ffill")

    pt_sl = params.get("pt_sl", labeling_config.pt_sl_ratio)
    
    events = create_events(
        close=df_aligned["close"],
        event_timestamps=target_vol.index,
        target_vol=target_vol,
        side=side,
        max_holding=labeling_config.max_holding_periods,
        pt_sl=pt_sl,
    )

    labels_df = get_labels(df_aligned["close"], events)
    if labels_df is None or labels_df.empty:
        logger.error("Nenhuma label gerada. Abortando.")
        sys.exit(1)

    labels_df = labels_df.dropna(subset=["label"])

    common_idx = features.index.intersection(labels_df.index)
    X = features.loc[common_idx]
    y = (labels_df.loc[common_idx, "label"] == 1).astype(int)

    logger.info(f"Amostras para treinamento: {len(X)}")

    if len(X) < 30:
        logger.warning("Poucas amostras. O modelo pode não generalizar.")

    # --- Treino no dataset completo ---
    max_depth = params.get("xgb_max_depth", ml_config.xgb_max_depth)
    model = MetaClassifier(
        n_estimators=150,
        max_depth=max_depth,
        use_xgboost=True,
    )
    weights = np.abs(labels_df.loc[y.index, "ret"])
    model.fit(X, y, sample_weight=weights)

    logger.success(f"Modelo treinado com {len(X)} amostras e {X.shape[1]} features.")

    return {
        "model": model,
        "optimal_d": optimal_d,
        "alpha": alpha,
        "feature_columns": list(X.columns),
    }


# ---------------------------------------------------------------------------
# 2. Serialização
# ---------------------------------------------------------------------------
def save_model(artifacts: dict, path: Path) -> None:
    """Salva os artefatos de treinamento em disco."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)
    logger.success(f"Modelo salvo em: {path}")


def load_model(path: Path) -> dict:
    """Carrega artefatos previamente salvos."""
    with open(path, "rb") as f:
        artifacts = pickle.load(f)
    logger.success(f"Modelo carregado de: {path}")
    return artifacts


# ---------------------------------------------------------------------------
# 3. Pipeline callable para o Engine
# ---------------------------------------------------------------------------
class LivePipeline:
    """
    Encapsula todo o pipeline de decisão num objeto callable.

    O AsyncTradingEngine chama `pipeline(df_snapshot)` a cada ciclo.
    Retorna um dict com side, meta_prob, kelly_fraction e price.
    """

    def __init__(self, artifacts: dict) -> None:
        self.model: MetaClassifier = artifacts["model"]
        self.optimal_d: float = artifacts["optimal_d"]
        self.alpha: TrendFollowingAlpha = artifacts["alpha"]
        self.feature_columns: list[str] = artifacts["feature_columns"]

    def __call__(self, df: pd.DataFrame) -> dict:
        """
        Recebe um snapshot histórico (ex: últimas 500 barras) e retorna
        a decisão de trading.
        """
        try:
            # 1. Features
            features = compute_all_features(df)
            features["ffd"] = frac_diff_ffd(df["close"], d=self.optimal_d)
            features = features.dropna()

            if features.empty or len(features) < 10:
                return self._neutral()

            # Garante mesmas colunas do treino
            missing = set(self.feature_columns) - set(features.columns)
            for col in missing:
                features[col] = 0.0
            features = features[self.feature_columns]

            # 2. Sinal Alpha (última barra)
            signal = self.alpha.generate_signal(df)
            current_side = int(signal.iloc[-1])

            if current_side == 0:
                return self._neutral()

            # 3. Meta-Modelo: probabilidade de sucesso
            last_features = features.iloc[[-1]]
            meta_prob = float(self.model.predict_proba(last_features)[:, 1][0])

            # 4. Kelly Sizing
            kelly_f = float(compute_kelly_fraction(meta_prob))

            price = float(df["close"].iloc[-1])

            return {
                "side": current_side,
                "meta_prob": meta_prob,
                "kelly_fraction": kelly_f,
                "price": price,
            }

        except Exception as e:
            logger.error(f"Erro no pipeline de predição: {e}")
            return self._neutral()

    @staticmethod
    def _neutral() -> dict:
        return {"side": 0, "meta_prob": 0.0, "kelly_fraction": 0.0, "price": 0.0}


# ---------------------------------------------------------------------------
# 4. Dados históricos para treinamento
# ---------------------------------------------------------------------------
def fetch_training_data(symbol: str, years: float, interval: str) -> pd.DataFrame:
    """Baixa dados do yfinance para treinar o modelo."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance não instalado. Execute: pip install yfinance")
        sys.exit(1)

    # Limites do Yahoo: 1h -> 730 dias, outros intraday -> 60 dias.
    # Usamos uma margem de segurança (729 e 59).
    end = pd.Timestamp.now()
    requested_days = int(years * 365)

    if interval in ["1h", "60m"]:
        days = min(requested_days, 729)
    elif interval in ["2m", "5m", "15m", "30m", "90m"]:
        days = min(requested_days, 59)
    elif interval == "1m":
        days = min(requested_days, 6)
    else:
        days = requested_days

    start = end - pd.Timedelta(days=days)

    logger.info(f"Baixando {days} dias de dados ({interval}) para {symbol}...")
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)

    if df.empty:
        logger.error(f"Nenhum dado para {symbol}.")
        sys.exit(1)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    logger.success(f"Dados de treino: {len(df)} barras ({interval})")
    return df

def fetch_mt5_training_data(symbol: str, interval: str, n_bars: int) -> pd.DataFrame:
    """Baixa dados do MT5 para treinar o modelo."""
    tf = INTERVAL_TO_TF.get(interval, 60)
    logger.info(f"Baixando {n_bars} barras ({interval}) do MT5 para {symbol}...")
    
    try:
        with mt5_session() as conn:
            df = extract_ohlc(symbol=symbol, timeframe=tf, n_bars=n_bars)
            
        if df.empty:
            logger.error(f"Nenhum dado encontrado no MT5 para {symbol}.")
            sys.exit(1)
            
        # O extractor já retorna time como index e colunas em minúsculas
        # Usa tick_volume como volume
        df = df[["open", "high", "low", "close", "tick_volume"]].rename(columns={"tick_volume": "volume"}).dropna()
        logger.success(f"Dados de treino MT5: {len(df)} barras ({interval})")
        return df
    except Exception as e:
        logger.error(f"Erro ao extrair dados de treino do MT5: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
async def run(pipeline: LivePipeline, symbols: list[str], max_position: float):
    """Inicia o motor assíncrono."""
    engine = AsyncTradingEngine(
        model_pipeline=pipeline,
        symbols=symbols,
        max_position=max_position,
    )
    try:
        await engine.run_forever()
    except KeyboardInterrupt:
        logger.warning("Interrupção manual (Ctrl+C).")
    finally:
        engine.stop()
        logger.info("Motor finalizado.")


def main():
    parser = argparse.ArgumentParser(
        description="TradeSystem5000 — Execução Live / Paper Trading"
    )
    parser.add_argument(
        "--symbol", type=str, default="PETR4.SA",
        help="Ativo a ser monitorado (padrão: PETR4.SA, .SA ignorado no MT5)",
    )
    parser.add_argument(
        "--data-source", type=str, default="mt5", choices=["mt5", "yfinance"],
        help="Fonte de dados para treino (padrão: mt5)",
    )
    parser.add_argument(
        "--interval", type=str, default="1h",
        choices=["1d", "1h", "15m", "5m", "1m"],
        help="Intervalo das barras para treino (padrão: 1h)",
    )
    parser.add_argument(
        "--years", type=float, default=2,
        help="Anos de histórico para treino (padrão: 2 para yfinance)",
    )
    parser.add_argument(
        "--n-bars", type=int, default=5000,
        help="Quantidade de barras para treino (padrão: 5000 para mt5)",
    )
    parser.add_argument(
        "--max-position", type=float, default=2.0,
        help="Posição máxima em lotes (padrão: 2.0)",
    )
    parser.add_argument(
        "--load-model", type=str, default=None,
        help="Caminho para carregar modelo pré-treinado (.pkl).",
    )
    parser.add_argument(
        "--force-optimize", action="store_true",
        help="Força re-otimização dos hiperparâmetros (ignora arquivo JSON existente).",
    )

    args = parser.parse_args()

    if args.data_source == "mt5":
        args.symbol = args.symbol.replace(".SA", "")

    logger.info("=" * 60)
    logger.info("  TradeSystem5000 — Paper / Live Execution")
    logger.info(f"  Modo: {execution_config.mode.upper()}")
    logger.info(f"  Datasource: {args.data_source.upper()}")
    logger.info(f"  Ativo: {args.symbol}")
    logger.info("=" * 60)

    # --- Inicializa MT5 se modo live ---
    if execution_config.mode == "live":
        import MetaTrader5 as mt5
        if not mt5.initialize():
            logger.critical(f"Falha ao inicializar MT5: {mt5.last_error()}")
            sys.exit(1)
        logger.success("MetaTrader 5 conectado.")

    # --- Auto-Otimização (Param Store) ---
    logger.info("Verificando parâmetros otimizados...")
    if not params_exist(args.symbol) or args.force_optimize:
        logger.info(f"Otimizando parâmetros para {args.symbol}...")
        
        # Otimização precisa dos dados
        if args.data_source == "mt5":
            df_opt = fetch_mt5_training_data(args.symbol, args.interval, args.n_bars)
        else:
            df_opt = fetch_training_data(args.symbol, args.years, args.interval)
            
        from src.optimization.tuner import run_optimization
        opt_results = run_optimization(df_opt, interval=args.interval)
        
        save_optimized_params(
            symbol=args.symbol, 
            params=opt_results["params"], 
            metadata=opt_results["metadata"]
        )
        logger.success("Parâmetros otimizados e salvos com sucesso!")
        
    store_data = load_optimized_params(args.symbol)
    optimized_params = store_data["params"] if store_data else None

    # --- Modelo ---
    model_path = Path("models") / f"model_{args.symbol}.pkl"

    if args.load_model:
        artifacts = load_model(Path(args.load_model))
    elif model_path.exists() and not args.force_optimize:
        logger.info(f"Modelo existente encontrado: {model_path}")
        artifacts = load_model(model_path)
    else:
        logger.info("Nenhum modelo encontrado (ou força-otimização). Iniciando treinamento...")
        if args.data_source == "mt5":
            df = fetch_mt5_training_data(args.symbol, args.interval, args.n_bars)
        else:
            df = fetch_training_data(args.symbol, args.years, args.interval)
        artifacts = train_model(df, interval=args.interval, params=optimized_params)
        save_model(artifacts, model_path)

    # --- Pipeline ---
    pipeline = LivePipeline(artifacts)

    # --- Execução ---
    asyncio.run(run(pipeline, symbols=[args.symbol], max_position=args.max_position))


if __name__ == "__main__":
    main()

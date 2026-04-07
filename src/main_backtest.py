"""
Orquestrador de Backtest do TradeSystem5000.
Conecta todas as fases: Dados -> Features -> Alpha/Labeling -> ML -> Backtest
"""

import argparse
import sys
import numpy as np
import pandas as pd
from loguru import logger
from config.settings import feature_config, labeling_config, ml_config

# Importa módulos das fases
from src.features.indicators import compute_all_features
from src.features.frac_diff import frac_diff_ffd, find_min_d
from src.features.cusum_filter import adaptive_cusum_events
from src.data.bar_sampler import volume_bars, dollar_bars
from src.labeling.alpha import TrendFollowingAlpha, get_signal_events
from src.labeling.volatility import get_volatility_targets
from src.labeling.triple_barrier import create_events, get_labels
from src.modeling.classifier import MetaClassifier
from src.backtest.metrics import performance_report
from src.backtest.cpcv import CombinatorialPurgedCV
from src.backtest.attribution import attribution_analysis, trade_level_attribution, attribution_summary

def generate_synthetic_data(n_days=1000):
    """Gera dados sintéticos de preços OHLCV (Random Walk) para demonstração."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="D")
    
    returns = np.random.normal(loc=0.0002, scale=0.015, size=n_days)
    close = 100 * np.exp(np.cumsum(returns))
    
    high = close * (1 + np.abs(np.random.normal(0, 0.005, size=n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, size=n_days)))
    
    open_price = pd.Series(close).shift(1).fillna(100).values
    high = np.maximum.reduce([high, open_price, close])
    low = np.minimum.reduce([low, open_price, close])
    
    volume = np.random.lognormal(mean=10, sigma=1, size=n_days)
    
    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)
    
    return df

def fetch_yfinance_data(symbol="PETR4.SA", years=5, interval="1d"):
    """Baixa dados históricos do Yahoo Finance com suporte a diferentes granularidades."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("A biblioteca 'yfinance' não está instalada. Execute: pip install yfinance")
        sys.exit(1)
        
    logger.info(f"Baixando dados do Yahoo Finance para {symbol} (Intervalo: {interval})...")
    
    # Ajusta o tempo de lookback se for intraday (yfinance tem limites rígidos)
    # 1h -> 730 dias, outros intraday -> 60 dias. Usamos margem de segurança (729/59).
    requested_days = int(years * 365)
    
    if interval in ["1h", "60m"]:
        days = min(requested_days, 729)
        if requested_days > 729:
            logger.warning("Intervalo 1h limitado a 730 dias no Yahoo Finance. Usando 729 dias.")
    elif interval in ["1m", "2m", "5m", "15m", "30m", "90m"]:
        max_d = 6 if interval == "1m" else 59
        days = min(requested_days, max_d)
        if requested_days > max_d:
            logger.warning(f"Intervalo {interval} limitado a {max_d+1} dias no Yahoo Finance. Usando {days} dias.")
    else:
        days = requested_days

    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=days)
    
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    
    if df.empty:
        logger.error(f"Nenhum dado encontrado para {symbol} no intervalo {interval}.")
        sys.exit(1)
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    logger.success(f"Dados baixados: {len(df)} barras ({interval}).")
    return df

def run_pipeline(df, interval="1d", use_volume_bars=False, params=None):
    """Executa o core do pipeline de backtest dado um DataFrame OHLCV."""
    
    if params is None:
        params = {}

    # ---------------------------------------------------------
    # Fase 1.2: Amostragem de Barras (Volume/Dollar)
    # ---------------------------------------------------------
    if use_volume_bars:
        logger.info("--- Fase 1.2: Aplicando Amostragem de Barras de Volume ---")
        # Tratamos OHLCV como "ticks" para o sampler
        ticks_pseudo = df.rename(columns={"close": "last"})
        df = volume_bars(ticks_pseudo, threshold=None) # Usa default da config
        interval = "volume"

    # Mapeamento de períodos por ano para métricas anualizadas (SR, etc.)
    periods_map = {
        "1d": 252,
        "1h": 252 * 7,      # Aproximação pregão B3
        "60m": 252 * 7,
        "15m": 252 * 7 * 4,
        "5m": 252 * 7 * 12,
        "volume": 252 * 7 * 6 # Estimativa para volume bars
    }
    ppy = periods_map.get(interval, 252)

    # ---------------------------------------------------------
    # Fase 2: Feature Engineering
    # ---------------------------------------------------------
    logger.info("--- Fase 2: Feature Engineering ---")
    features = compute_all_features(df)
    
    # Fase 3.1: Otimização de Diferenciação Fracionária
    logger.info("Fase 3.1: Otimizando d para FFD...")
    optimal_d = find_min_d(df["close"])
    features["ffd"] = frac_diff_ffd(df["close"], d=optimal_d)
    
    features = features.dropna()
    df = df.reindex(features.index)
    logger.info(f"Features prontas: {features.shape}")
    
    # ---------------------------------------------------------
    # Fase 3: Alpha e Labeling (Tripla Barreira)
    # ---------------------------------------------------------
    logger.info("--- Fase 3: Alpha e Labeling ---")
    
    # Extrai spans dos params ou usa os padrões do config
    fast_span = params.get("alpha_fast", labeling_config.trend_fast_span)
    slow_span = params.get("alpha_slow", labeling_config.trend_slow_span)
    
    alpha_model = TrendFollowingAlpha(fast_span=fast_span, slow_span=slow_span)
    signal = alpha_model.generate_signal(df)
    
    signal_events = get_signal_events(signal)
    
    # Filtro CUSUM (Fase 1 do plano de otimização)
    if "cusum_threshold" in params or hasattr(feature_config, "cusum_threshold_pct"):
        threshold = params.get("cusum_threshold", feature_config.cusum_threshold_pct)
        cusum_ts = adaptive_cusum_events(df["close"], threshold_multiplier=threshold)
        signal_events = signal_events.intersection(cusum_ts)
        logger.info("Eventos após filtro CUSUM: {}", len(signal_events))

    if len(signal_events) == 0:
        logger.error("Nenhum sinal gerado pelo Alpha Model. Tente ajustar os parâmetros.")
        return None
        
    target_vol = get_volatility_targets(df["close"], signal_events, span=labeling_config.vol_span)
    
    pt_sl = params.get("pt_sl", labeling_config.pt_sl_ratio)
    
    events = create_events(
        close=df["close"],
        event_timestamps=target_vol.index,
        target_vol=target_vol,
        side=signal.loc[target_vol.index],
        max_holding=20,
        pt_sl=pt_sl
    )
    
    labels_df = get_labels(df["close"], events)
    if labels_df is None or labels_df.empty:
        logger.error("Nenhuma label gerada. Encerrando.")
        return None
        
    labels_df = labels_df.dropna(subset=["label"])
    
    common_idx = features.index.intersection(labels_df.index)
    X = features.loc[common_idx]
    y_meta = (labels_df.loc[common_idx, "label"] == 1).astype(int)
    
    # ---------------------------------------------------------
    # Fase 4: Machine Learning (Classificador Secundário com CV Purificada)
    # ---------------------------------------------------------
    logger.info("--- Fase 4: Meta-Model com Validação Cruzada Purificada (CPCV) ---")
    
    if len(X) < 20:
        logger.warning(f"Amostragem muito pequena ({len(X)} eventos).")

    # Amostras info para Purga e Embargo (t0 -> t1)
    samples_info = events.loc[common_idx, "t1"]
    
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, samples_info=samples_info)
    splits = cv.split(X)
    
    all_test_preds = []
    all_test_probs = []
    all_test_labels = []
    all_train_sharpes = []

    # Params para o MetaClassifier
    max_depth = params.get("xgb_max_depth", ml_config.xgb_max_depth)

    for i, (train_idx, test_idx) in enumerate(splits):
        if len(train_idx) < 10: continue
        
        X_train, y_train = X.iloc[train_idx], y_meta.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y_meta.iloc[test_idx]
        
        model = MetaClassifier(n_estimators=150, max_depth=max_depth, use_xgboost=True)
        
        weights = np.abs(labels_df.loc[X_train.index, "ret"])
        model.fit(X_train, y_train, sample_weight=weights)
        
        # Sharpe de treino para diagnóstico
        train_probs = model.predict_proba(X_train)[:, 1]
        train_preds = model.predict(X_train)
        train_labels = labels_df.loc[X_train.index].copy()
        
        train_trades = pd.DataFrame({
            "ret": train_labels["ret"],
            "side": train_labels["side"],
            "meta_label": train_preds,
            "bet_size": train_probs
        }, index=X_train.index)
        
        train_attr = trade_level_attribution(train_trades)
        train_sr = attribution_analysis(train_attr["net_return"], train_attr["alpha_contribution"], periods_per_year=ppy)["sharpe_full_system"]
        all_train_sharpes.append(train_sr)
        
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        
        all_test_preds.append(pd.Series(preds, index=X_test.index))
        all_test_probs.append(pd.Series(probs, index=X_test.index))
        all_test_labels.append(y_test)

    if not all_test_preds:
        logger.error("Falha ao gerar folds de validação.")
        return None

    avg_train_sharpe = np.mean(all_train_sharpes) if all_train_sharpes else 0.0

    # Consolida resultados do teste
    y_pred_all = pd.concat(all_test_preds).groupby(level=0).mean() > 0.5
    y_prob_all = pd.concat(all_test_probs).groupby(level=0).mean()
    y_test_all = pd.concat(all_test_labels).groupby(level=0).first()
    
    # ---------------------------------------------------------
    # Fase 5: Backtest, Métricas e Atribuição
    # ---------------------------------------------------------
    logger.info("--- Fase 5: Métricas e Atribuição de Performance ---")
    
    test_labels = labels_df.loc[y_test_all.index].copy()
    
    trades = pd.DataFrame({
        "ret": test_labels["ret"],
        "side": test_labels["side"],
        "meta_label": y_pred_all.astype(int),
        "bet_size": y_prob_all,
        "cost": 0.0005 
    }, index=y_test_all.index)
    
    trade_attr = trade_level_attribution(trades)
    
    net_returns = trade_attr["net_return"]
    alpha_returns = trade_attr["alpha_contribution"]
    
    logger.info(">>> Performance Final (OOS - Out-of-Sample) <<<")
    performance_report(net_returns, periods_per_year=ppy)
    
    logger.info(">>> Análise de Atribuição (Sharpe Lift) <<<")
    attr_results = attribution_analysis(net_returns, alpha_returns, periods_per_year=ppy)
    summary = attribution_summary(trade_attr)
    
    if attr_results["sharpe_lift_meta"] > 0:
        logger.success(f"Meta-Model agregou valor! Sharpe Lift: {attr_results['sharpe_lift_meta']:.2f}")
    else:
        logger.warning(f"Meta-Model não superou o Alpha. Sharpe Lift: {attr_results['sharpe_lift_meta']:.2f}")
    
    logger.success("Pipeline orquestrado com sucesso!")

    return {
        "sharpe": attr_results["sharpe_full_system"],
        "sharpe_alpha": attr_results["sharpe_alpha_only"],
        "sharpe_lift": attr_results["sharpe_lift_meta"],
        "sharpe_train": avg_train_sharpe,
        "n_trades": len(trades),
        "net_returns": net_returns,
        "alpha_returns": alpha_returns
    }

def main():
    parser = argparse.ArgumentParser(description="TradeSystem5000 Backtest Orchestrator")
    parser.add_argument("--mode", choices=["synthetic", "yfinance", "real"], default="yfinance", 
                        help="Modo de dados. Padrão: yfinance.")
    parser.add_argument("--symbol", type=str, default="PETR4.SA", help="Ativo (ex: PETR4.SA).")
    parser.add_argument("--years", type=float, default=2, help="Anos de história (limite 2 p/ 1h em yfinance).")
    parser.add_argument("--interval", type=str, choices=["1d", "1h", "15m", "5m"], default="1h", 
                        help="Intervalo das barras (ex: 1d, 1h).")
    parser.add_argument("--volume-bars", action="store_true", help="Usa amostragem de barras de volume (Fase 1.2).")
    
    args = parser.parse_args()
    
    if args.mode == "synthetic":
        logger.info("Iniciando Modo SINTÉTICO (2000 dias)...")
        df = generate_synthetic_data(n_days=2000)
    elif args.mode == "yfinance":
        if args.volume_bars and args.interval == "1h":
            logger.info("Volume Bars solicitado: Alterando intervalo yfinance para 5m para melhor precisão.")
            args.interval = "5m"
            args.years = min(args.years, 0.16)
            
        logger.info(f"Iniciando Modo YFINANCE (Ativo: {args.symbol}, Intervalo: {args.interval})...")
        df = fetch_yfinance_data(symbol=args.symbol, years=args.years, interval=args.interval)
    else:
        logger.error("Modo 'real' com MT5 será desenvolvido na Fase 6.")
        sys.exit(1)
        
    run_pipeline(df, interval=args.interval, use_volume_bars=args.volume_bars)

if __name__ == "__main__":
    main()

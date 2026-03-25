"""
Orquestrador de Backtest do TradeSystem5000.
Conecta todas as fases: Dados -> Features -> Alpha/Labeling -> ML -> Backtest
"""

import argparse
import sys
import numpy as np
import pandas as pd
from loguru import logger

# Importa módulos das fases
from src.features.indicators import compute_all_features
from src.features.frac_diff import frac_diff_ffd
from src.labeling.alpha import TrendFollowingAlpha, get_signal_events
from src.labeling.volatility import get_volatility_targets
from src.labeling.triple_barrier import create_events, get_labels
from src.modeling.classifier import MetaClassifier
from src.backtest.metrics import performance_report
from src.backtest.attribution import attribution_analysis, trade_level_attribution

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
    if interval in ["1h", "60m"]:
        if years > 2:
            logger.warning("Intervalo 1h limitado a 2 anos (730 dias) no Yahoo Finance. Ajustando...")
            years = 2
    elif interval in ["1m", "2m", "5m", "15m", "30m", "90m"]:
        if years > 0.16:
            logger.warning(f"Intervalo {interval} limitado a 60 dias no Yahoo Finance. Ajustando...")
            years = 0.16 # ~60 dias

    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=int(years * 365))
    
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

def run_pipeline(df, interval="1d"):
    """Executa o core do pipeline de backtest dado um DataFrame OHLCV."""
    
    # Mapeamento de períodos por ano para métricas anualizadas (SR, etc.)
    periods_map = {
        "1d": 252,
        "1h": 252 * 7,      # Aproximação pregão B3
        "60m": 252 * 7,
        "15m": 252 * 7 * 4,
        "5m": 252 * 7 * 12
    }
    ppy = periods_map.get(interval, 252)

    # ---------------------------------------------------------
    # Fase 2: Feature Engineering
    # ---------------------------------------------------------
    logger.info("--- Fase 2: Feature Engineering ---")
    features = compute_all_features(df)
    
    # Diferenciação Fracionária
    # Nota: Com janelas de pesos muito grandes em intraday (muitos dados), pode ser pesado.
    features["ffd"] = frac_diff_ffd(df["close"], d=0.4)
    
    features = features.dropna()
    df = df.reindex(features.index)
    logger.info(f"Features prontas: {features.shape}")
    
    # ---------------------------------------------------------
    # Fase 3: Alpha e Labeling (Tripla Barreira)
    # ---------------------------------------------------------
    logger.info("--- Fase 3: Alpha e Labeling ---")
    alpha_model = TrendFollowingAlpha(fast_span=10, slow_span=50)
    signal = alpha_model.generate_signal(df)
    
    signal_events = get_signal_events(signal)
    
    if len(signal_events) == 0:
        logger.error("Nenhum sinal gerado pelo Alpha Model. Tente ajustar os parâmetros.")
        return
        
    target_vol = get_volatility_targets(df["close"], signal_events, span=50)
    
    events = create_events(
        close=df["close"],
        event_timestamps=target_vol.index,
        target_vol=target_vol,
        side=signal.loc[target_vol.index],
        max_holding=20
    )
    
    labels_df = get_labels(df["close"], events)
    if labels_df is None or labels_df.empty:
        logger.error("Nenhuma label gerada. Encerrando.")
        return
        
    labels_df = labels_df.dropna(subset=["label"])
    
    common_idx = features.index.intersection(labels_df.index)
    X = features.loc[common_idx]
    y_meta = (labels_df.loc[common_idx, "label"] == 1).astype(int)
    
    # ---------------------------------------------------------
    # Fase 4: Machine Learning (Classificador Secundário)
    # ---------------------------------------------------------
    logger.info("--- Fase 4: Classificador Secundário (Meta-Model) ---")
    
    if len(X) < 10:
        logger.warning(f"Amostragem muito pequena ({len(X)} eventos). ML pode não generalizar.")
        
    split_idx = int(len(X) * 0.7)
    train_idx = X.index[:split_idx]
    test_idx = X.index[split_idx:]
    
    X_train, y_train = X.loc[train_idx], y_meta.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y_meta.loc[test_idx]
    
    train_mask = X_train.notna().all(axis=1) & y_train.notna()
    test_mask  = X_test.notna().all(axis=1) & y_test.notna()
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    logger.info(f"Treino: {len(X_train)} eventos | Teste: {len(X_test)} eventos")
    
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Sem dados suficientes para Treino/Teste após limpeza.")
        return

    meta_model = MetaClassifier(n_estimators=100, max_depth=3)
    meta_model.fit(X_train, y_train)
    
    meta_model.evaluate(X_test, y_test)
    prob_predictions = meta_model.predict_proba(X_test)[:, 1]
    meta_predictions = meta_model.predict(X_test)
    
    # ---------------------------------------------------------
    # Fase 5: Backtest, Métricas e Atribuição
    # ---------------------------------------------------------
    logger.info("--- Fase 5: Métricas e Atribuição de Performance ---")
    test_labels = labels_df.loc[test_idx[test_mask]].copy()
    
    trades = pd.DataFrame({
        "ret": test_labels["ret"],
        "side": test_labels["side"],
        "meta_label": meta_predictions,
        "bet_size": prob_predictions,
        "cost": 0.0005 
    }, index=test_idx[test_mask])
    
    trade_attr = trade_level_attribution(trades)
    
    net_returns = trade_attr["net_return"]
    alpha_returns = trade_attr["alpha_contribution"]
    
    logger.info(">>> Performance Final (Conjunto de Teste) <<<")
    performance_report(net_returns, periods_per_year=ppy)
    
    logger.info(">>> Análise de Atribuição <<<")
    attribution_analysis(net_returns, alpha_returns, periods_per_year=ppy)
    
    logger.success("Pipeline orquestrado com sucesso!")

def main():
    parser = argparse.ArgumentParser(description="TradeSystem5000 Backtest Orchestrator")
    parser.add_argument("--mode", choices=["synthetic", "yfinance", "real"], default="yfinance", 
                        help="Modo de dados. Padrão: yfinance.")
    parser.add_argument("--symbol", type=str, default="PETR4.SA", help="Ativo (ex: PETR4.SA).")
    parser.add_argument("--years", type=float, default=2, help="Anos de história (limite 2 p/ 1h em yfinance).")
    parser.add_argument("--interval", type=str, choices=["1d", "1h", "15m", "5m"], default="1h", 
                        help="Intervalo das barras (ex: 1d, 1h). Padrão alterado para 1h.")
    
    args = parser.parse_args()
    
    if args.mode == "synthetic":
        logger.info("Iniciando Modo SINTÉTICO (2000 dias)...")
        df = generate_synthetic_data(n_days=2000)
    elif args.mode == "yfinance":
        logger.info(f"Iniciando Modo YFINANCE (Ativo: {args.symbol}, Intervalo: {args.interval})...")
        df = fetch_yfinance_data(symbol=args.symbol, years=args.years, interval=args.interval)
    else:
        logger.error("Modo 'real' com MT5 será desenvolvido na Fase 6.")
        sys.exit(1)
        
    run_pipeline(df, interval=args.interval)

if __name__ == "__main__":
    main()

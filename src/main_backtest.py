"""
Orquestrador de Backtest — TradeSystem5000.

Este módulo integra todas as fases da metodologia AFML (Advances in Financial Machine Learning):
1.  **Dados**: Extração de fontes como MT5 ou Yahoo Finance.
2.  **Amostragem**: Conversão para barras de volume ou dólar.
3.  **Features**: Cálculo de indicadores técnicos e Diferenciação Fracionária (FFD).
4.  **Labeling**: Triple Barrier Method e Meta-Labeling.
5.  **ML**: Treinamento de Meta-Classificador e validação via Combinatorial Purged CV.
6.  **Backtest**: Simulação, atribuição de performance e métricas estatísticas.

Argumentos
----------
--mode : {'synthetic', 'yfinance', 'mt5'}
    Origem dos dados. Default: 'mt5'.
--symbol : str
    Ticker do ativo (ex: PETR4.SA, WINJ26).
--years : float
    Anos de histórico (para yfinance/mt5). Default: 2.
--n-bars : int
    Quantidade de barras (para mt5). Default: 5000.
--interval : {'1d', '1h', '15m', '5m', '1m'}
    Timeframe das barras. Default: '1h'.
--volume-bars : bool
    Se presente, utiliza amostragem por barras de volume (AFML Cap. 2).

Exemplos de Uso
---------------
$ python -m src.main_backtest --mode yfinance --symbol PETR4.SA --interval 1d
$ python -m src.main_backtest --mode mt5 --symbol WINJ26 --interval 5m --n-bars 10000
$ python -m src.main_backtest --mode mt5 --symbol PETR4 --volume-bars

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
"""

import argparse
import sys

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import FeatureConfig, feature_config, labeling_config, ml_config
from src.backtest.attribution import (
    attribution_analysis,
    attribution_summary,
    trade_level_attribution,
)
from src.backtest.cpcv import CombinatorialPurgedCV
from src.backtest.metrics import performance_report
from src.data.bar_sampler import volume_bars
from src.data.extractor import INTERVAL_TO_TF, extract_ohlc

# Importa módulos das fases
from src.data.mt5_connector import mt5_session
from src.features.cusum_filter import adaptive_cusum_events
from src.features.frac_diff import find_min_d, frac_diff_ffd
from src.features.indicators import compute_all_features
from src.labeling.alpha import TrendFollowingAlpha, get_signal_events
from src.labeling.triple_barrier import create_events, get_labels
from src.labeling.volatility import get_volatility_targets
from src.modeling.classifier import MetaClassifier
from src.modeling.feature_evaluation import evaluate_features_mda, evaluate_features_shap


def generate_synthetic_data(n_days: int = 1000) -> pd.DataFrame:
    """
    Gera dados sintéticos de preços OHLCV (Random Walk) para demonstração.

    Parameters
    ----------
    n_days : int, optional
        Número de dias de dados a gerar. Default: 1000.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas [open, high, low, close, volume] e DateTimeIndex.
    """
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

def fetch_yfinance_data(symbol: str = "PETR4.SA", years: float = 5, interval: str = "1d") -> pd.DataFrame:
    """
    Baixa dados históricos do Yahoo Finance com suporte a diferentes granularidades.

    Ajusta automaticamente o período de lookback para respeitar os limites da API
    do Yahoo Finance para dados intraday.

    Parameters
    ----------
    symbol : str, optional
        Ticker do ativo. Default: "PETR4.SA".
    years : float, optional
        Anos de história desejados. Default: 5.
    interval : str, optional
        Granularidade das barras (ex: 1m, 5m, 1h, 1d). Default: "1d".

    Returns
    -------
    pd.DataFrame
        DataFrame OHLCV limpo.
    """
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

def fetch_mt5_data(symbol: str = "PETR4", years: float = 5, interval: str = "1h", n_bars: int = 5000) -> pd.DataFrame:
    """
    Baixa dados históricos do MetaTrader 5 usando os módulos de dados internos.

    Parameters
    ----------
    symbol : str, optional
        Ativo no MT5. Default: "PETR4".
    years : float, optional
        Anos de história (utilizado se n_bars for ignorado). Default: 5.
    interval : str, optional
        Granularidade (1m, 5m, 1h, etc). Default: "1h".
    n_bars : int, optional
        Número de barras a solicitar. Default: 5000.

    Returns
    -------
    pd.DataFrame
        DataFrame OHLCV com volumes normalizados.
    """
    logger.info(f"Baixando dados do MT5 para {symbol} (Intervalo: {interval})...")

    tf = INTERVAL_TO_TF.get(interval, 60)

    try:
        with mt5_session() as conn:
            df = extract_ohlc(symbol=symbol, timeframe=tf, n_bars=n_bars)

        if df.empty:
            logger.error(f"Nenhum dado encontrado no MT5 para {symbol}.")
            sys.exit(1)

        # O extractor já retorna time como index e colunas em minúsculas
        # Usa tick_volume como volume (comum no backtest MT5)
        df = df[["open", "high", "low", "close", "tick_volume"]].rename(columns={"tick_volume": "volume"}).dropna()
        logger.success(f"Dados do MT5 baixados: {len(df)} barras ({interval}).")
        return df
    except Exception as e:
        logger.error(f"Erro ao extrair dados do MT5: {e}")
        sys.exit(1)

def run_pipeline(df: pd.DataFrame, interval: str = "1d", use_volume_bars: bool = False, params: dict | None = None) -> dict:
    """
    Executa o core do pipeline de backtest dado um DataFrame OHLCV.

    Implementa a sequência completa de processamento AFML: Amostragem -> Features ->
    FFD -> Labeling -> Meta-Labeling -> Avaliação.

    Parameters
    ----------
    df : pd.DataFrame
        Dados históricos OHLCV.
    interval : str, optional
        Granularidade temporal original. Default: "1d".
    use_volume_bars : bool, optional
        Se True, aplica amostragem por volume antes do processamento. Default: False.
    params : dict, optional
        Dicionário de hiperparâmetros (fast_span, slow_span, pt_sl, etc).

    Returns
    -------
    dict
        Dicionário contendo os resultados de performance, atribuição e dados processados.
    """

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

    feat_keys = [
        "atr_period",
        "zscore_window",
        "ma_dist_fast_period",
        "ma_dist_slow_period",
        "moments_window",
    ]
    feat_overrides = {k: params[k] for k in feat_keys if k in params}

    if feat_overrides:
        from dataclasses import asdict

        base = asdict(feature_config)
        base.update(feat_overrides)
        dynamic_config = FeatureConfig(**base)
    else:
        dynamic_config = feature_config

    features = compute_all_features(df, config=dynamic_config)

    # Fase 3.1: Otimização de Diferenciação Fracionária
    if "ffd_d" in params:
        logger.info(f"Fase 3.1: Usando d={params['ffd_d']:.4f} otimizado para FFD...")
        features["ffd"] = frac_diff_ffd(df["close"], d=params["ffd_d"])
    else:
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
    be_trigger = params.get("be_trigger", 0.0)

    events = create_events(
        close=df["close"],
        event_timestamps=target_vol.index,
        target_vol=target_vol,
        side=signal.loc[target_vol.index],
        max_holding=20,
        pt_sl=pt_sl
    )

    labels_df = get_labels(df["close"], events, be_trigger=be_trigger, open_prices=df["open"])
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
    gamma = params.get("xgb_gamma", ml_config.xgb_gamma)
    min_child_weight = params.get("xgb_min_child_weight", ml_config.xgb_min_child_weight)
    reg_lambda = params.get("xgb_lambda", ml_config.xgb_lambda)
    reg_alpha = params.get("xgb_alpha", ml_config.xgb_alpha)
    meta_threshold = params.get("meta_threshold", 0.5)

    for i, (train_idx, test_idx) in enumerate(splits):
        if len(train_idx) < 10: continue

        X_train, y_train = X.iloc[train_idx], y_meta.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y_meta.iloc[test_idx]

        model = MetaClassifier(
            n_estimators=150,
            max_depth=max_depth,
            gamma=gamma,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            use_xgboost=True
        )

        weights = np.abs(labels_df.loc[X_train.index, "ret"])
        # Normaliza pesos para média 1.0 (min_child_weight do XGB opera sobre a Hessiana = w * p*(1-p))
        w_mean = weights.mean()
        if w_mean > 0:
            weights = weights / w_mean
        model.fit(X_train, y_train, sample_weight=weights)

        # Sharpe de treino para diagnóstico
        train_probs = model.predict_proba(X_train)[:, 1]
        train_preds = train_probs > meta_threshold
        train_labels = labels_df.loc[X_train.index].copy()

        train_trades = pd.DataFrame({
            "ret": train_labels["ret"],
            "side": train_labels["side"],
            "meta_label": train_preds.astype(int),
            "bet_size": train_probs
        }, index=X_train.index)

        train_attr = trade_level_attribution(train_trades)
        train_sr = attribution_analysis(train_attr["net_return"], train_attr["alpha_contribution"], periods_per_year=ppy)["sharpe_full_system"]
        all_train_sharpes.append(train_sr)

        probs = model.predict_proba(X_test)[:, 1]
        preds = probs > meta_threshold

        all_test_preds.append(pd.Series(preds, index=X_test.index))
        all_test_probs.append(pd.Series(probs, index=X_test.index))
        all_test_labels.append(y_test)

    if not all_test_preds:
        logger.error("Falha ao gerar folds de validação.")
        return None

    avg_train_sharpe = np.mean(all_train_sharpes) if all_train_sharpes else 0.0

    # Consolida resultados do teste
    y_prob_all = pd.concat(all_test_probs).groupby(level=0).mean()
    y_pred_all = y_prob_all > meta_threshold
    y_test_all = pd.concat(all_test_labels).groupby(level=0).first()

    # Auditoria de Features (Apenas na execução final, não em trials do Optuna para poupar tempo)
    # Identificamos a execução final quando 'params' contém chaves de busca completa ou é chamado manualmente
    if params and "ma_dist_fast_period" in params:
        logger.info("--- Auditoria de Features Final (Modelo Global) ---")
        # Treinamos um modelo no dataset completo apenas para explicabilidade
        full_model = MetaClassifier(
            n_estimators=150,
            max_depth=max_depth,
            gamma=gamma,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            use_xgboost=True
        )
        full_weights = np.abs(labels_df.loc[X.index, "ret"])
        fw_mean = full_weights.mean()
        if fw_mean > 0:
            full_weights = full_weights / fw_mean
        full_model.fit(X, y_meta, sample_weight=full_weights)

        evaluate_features_shap(full_model, X, y_meta)
        evaluate_features_mda(full_model, X, y_meta)

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
        "cost": 40.0 / df.loc[y_test_all.index, "close"]  # Teste A: Slippage Ganancioso (40 pontos no WIN)
    }, index=y_test_all.index)

    trade_attr = trade_level_attribution(trades)

    net_returns = trade_attr["net_return"]
    alpha_returns = trade_attr["alpha_contribution"]

    logger.info(">>> Performance Final (OOS - Out-of-Sample) <<<")
    from src.backtest.metrics import calmar_ratio
    perf = performance_report(net_returns, periods_per_year=ppy)
    calmar = calmar_ratio(net_returns, periods_per_year=ppy)

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
        "calmar_ratio": calmar,
        "n_trades": len(trades),
        "net_returns": net_returns,
        "alpha_returns": alpha_returns
    }

def main():
    parser = argparse.ArgumentParser(description="TradeSystem5000 Backtest Orchestrator")
    parser.add_argument("--mode", choices=["synthetic", "yfinance", "mt5"], default="mt5",
                        help="Modo de dados. Padrão: mt5.")
    parser.add_argument("--symbol", type=str, default="PETR4.SA", help="Ativo (ex: PETR4.SA). (Sufixo .SA ignorado no MT5)")
    parser.add_argument("--years", type=float, default=2, help="Anos de história (limite 2 p/ 1h em yfinance).")
    parser.add_argument("--n-bars", type=int, default=5000, help="Quantidade de barras para MT5.")
    parser.add_argument("--interval", type=str, choices=["1d", "1h", "15m", "5m", "1m"], default="1h",
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
    elif args.mode == "mt5":
        symbol = args.symbol.replace(".SA", "")
        logger.info(f"Iniciando Modo MT5 (Ativo: {symbol}, Intervalo: {args.interval})...")
        df = fetch_mt5_data(symbol=symbol, years=args.years, interval=args.interval, n_bars=args.n_bars)
    else:
        logger.error("Modo inválido.")
        sys.exit(1)

    run_pipeline(df, interval=args.interval, use_volume_bars=args.volume_bars)

if __name__ == "__main__":
    main()

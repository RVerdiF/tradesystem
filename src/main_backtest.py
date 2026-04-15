"""Orquestrador de Backtest — TradeSystem5000.

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

from config.settings import FeatureConfig, cost_config, feature_config, labeling_config, ml_config
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
from src.features.order_flow import compute_vir, compute_vir_zscore
from src.labeling.alpha import CompositeAlpha, get_signal_events
from src.labeling.triple_barrier import create_events, get_labels
from src.labeling.volatility import get_volatility_targets
from src.modeling.classifier import MetaClassifier
from src.modeling.feature_evaluation import evaluate_features_mda, evaluate_features_shap


def generate_synthetic_data(n_days: int = 1000) -> pd.DataFrame:
    """Gera dados sintéticos de preços OHLCV (Random Walk) para demonstração.

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
    """Baixa dados históricos do Yahoo Finance com suporte a diferentes granularidades.

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
    """Baixa dados históricos do MetaTrader 5 usando os módulos de dados internos.

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
    """Executa o core do pipeline de backtest dado um DataFrame OHLCV.

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
    if len(df) == 0:
        logger.error("DataFrame vazio. Abortando run_pipeline.")
        return None

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

    features = compute_all_features(
        df,
        config=dynamic_config,
        is_volume_clock=use_volume_bars
    )

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
    # Adiciona série FracDiff para o Alpha operar sobre dados estacionários
    df = df.copy()
    df["close_fracdiff"] = features["ffd"]

    # Injeta features no df para uso no CompositeAlpha
    for col in features.columns:
        if col not in df.columns:
            df[col] = features[col]

    # Component 1 (Plan: FracDiff Signals): Log close_fracdiff injection explicitly
    logger.info(
        "close_fracdiff injectado em df: {} valores não-NaN de {} total",
        df["close_fracdiff"].notna().sum(),
        len(df),
    )

    # Component 3 (Plan: FracDiff Signals): Defensive assertion to catch NaN alignment bugs
    assert df["close_fracdiff"].notna().all(), (
        "close_fracdiff has NaN after reindex — check FFD computation or data alignment"
    )

    # Validação anti look-ahead: compute_all_features não produz colunas _zscore (sem
    # normalização explícita), então não há superfície de vazamento aqui. Se no futuro
    # normalize_features() for adicionado ao pipeline, capturar raw_features ANTES de
    # normalizar e chamar validate_no_lookahead(normalized=features[zscore_cols],
    # original=raw_features[base_cols]) — NÃO passar o mesmo df para ambos os argumentos.

    # ---------------------------------------------------------
    # Fase 3: Alpha e Labeling (Tripla Barreira)
    # ---------------------------------------------------------
    logger.info("--- Fase 3: Alpha e Labeling ---")

    # Extrai spans dos params ou usa os padrões do config
    long_fast_span = params.get("long_alpha_fast", labeling_config.long_fast_span)
    long_slow_span = params.get("long_alpha_slow", labeling_config.long_slow_span)
    short_fast_span = params.get("short_alpha_fast", labeling_config.short_fast_span)
    short_slow_span = params.get("short_alpha_slow", labeling_config.short_slow_span)
    long_hurst_threshold = params.get("long_hurst_threshold", feature_config.long_hurst_threshold)
    short_hurst_threshold = params.get("short_hurst_threshold", feature_config.short_hurst_threshold)
    long_voi_threshold = params.get("long_voi_threshold", feature_config.long_vol_imbalance_z_threshold)
    short_voi_threshold = params.get("short_voi_threshold", feature_config.short_vol_imbalance_z_threshold)

    alpha_model = CompositeAlpha(
        long_fast_span=long_fast_span,
        long_slow_span=long_slow_span,
        short_fast_span=short_fast_span,
        short_slow_span=short_slow_span,
        long_hurst_threshold=long_hurst_threshold,
        short_hurst_threshold=short_hurst_threshold,
        long_vir_zscore_threshold=long_voi_threshold,
        short_vir_zscore_threshold=short_voi_threshold,
    )
    signal = alpha_model.generate_signal(df)

    signal_events = get_signal_events(signal)

    # Filtro CUSUM (Fase 1 do plano de otimização)
    if "cusum_threshold" in params or hasattr(feature_config, "cusum_threshold_pct"):
        threshold = params.get("cusum_threshold", feature_config.cusum_threshold_pct)
        cusum_ts = adaptive_cusum_events(df["close"], threshold_multiplier=threshold)
        signal_events = signal_events.intersection(cusum_ts)
        logger.info("Eventos após filtro CUSUM: {}", len(signal_events))

    # --- VIR (Volume Imbalance Ratio) Filter ---
    # Applied AFTER CUSUM intersection and AFTER Hurst filter (if enabled).
    # NOT applied inside generate_signal() to avoid double-modification of alpha.py.
    voi_threshold = params.get("voi_threshold", None)
    voi_window = params.get("voi_window", 20)
    vir_filter_rate = None

    if voi_threshold is not None:
        logger.info(
            "VIR filter enabled: voi_window={}, voi_threshold={:.2f}",
            voi_window,
            voi_threshold,
        )

        # Compute VIR and rolling zscore on the full df
        vir = compute_vir(df, window=voi_window)
        vir_zscore = compute_vir_zscore(vir, window=voi_window)

        # Defensive: verify no lookahead — vir_zscore uses data up to and
        # including bar t (bar close). We only act at bar close. No lookahead.
        n_events_before_vir = len(signal_events)

        # Get side information from alpha signal
        event_sides = signal.loc[signal_events] if isinstance(signal_events, pd.DatetimeIndex) else signal.loc[signal_events.index]

        # Reindex vir_zscore to event timestamps; missing → NaN (event not filtered out)
        vir_at_events = vir_zscore.reindex(signal_events)
        sides_at_events = event_sides.reindex(signal_events)

        # Keep event if: (vir_zscore * side > voi_threshold) OR vir_zscore is NaN
        # NaN is treated as "no information" → do not filter out (conservative)
        vir_directional = vir_at_events * sides_at_events
        keep_mask = vir_directional.isna() | (vir_directional > voi_threshold)

        signal_events = signal_events[keep_mask.values]

        n_events_after_vir = len(signal_events)
        vir_removed = n_events_before_vir - n_events_after_vir
        vir_filter_rate = (
            vir_removed / n_events_before_vir if n_events_before_vir > 0 else 0.0
        )

        logger.info(
            "VIR filter: removed {}/{} events (filter_rate={:.1f}%)",
            vir_removed,
            n_events_before_vir,
            vir_filter_rate * 100,
        )
    else:
        vir_zscore = None

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

    labels_df = get_labels(
        df["close"],
        events,
        be_trigger=be_trigger,
        open_prices=df["open"],
        high_prices=df["high"],
        low_prices=df["low"],
    )
    if labels_df is None or labels_df.empty:
        logger.error("Nenhuma label gerada. Encerrando.")
        return None

    labels_df = labels_df.dropna(subset=["label"])

    # --- Component 4: Add VIR zscore feature for Meta-Model ---
    if vir_zscore is not None:
        # Align vir_zscore to the features index (already dropna'd from other features)
        features["vir_zscore"] = vir_zscore.reindex(features.index)

        n_vir_nonnull = features["vir_zscore"].notna().sum()
        n_features_total = len(features)
        logger.info(
            "Added vir_zscore to feature matrix: {} / {} non-NaN values ({:.1f}%).",
            n_vir_nonnull,
            n_features_total,
            100.0 * n_vir_nonnull / n_features_total if n_features_total > 0 else 0.0,
        )

        if n_vir_nonnull < 0.5 * n_features_total:
            logger.warning(
                "vir_zscore has >50% NaN in feature matrix. "
                "Check voi_window vs. available data length."
            )
    else:
        # VIR filter disabled; do not add feature (avoids NaN column in XGBoost)
        logger.debug("vir_zscore not added to features: VIR filter disabled (voi_threshold=None).")

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

    # Diagnóstico global de desbalanceamento e sanidade do min_child_weight
    global_pos_rate = y_meta.mean()
    logger.info(
        "Meta-Model dataset: {} amostras | {:.1f}% positivos ({} de {}) | min_child_weight={:.1f}",
        len(y_meta), global_pos_rate * 100, int(y_meta.sum()), len(y_meta), min_child_weight,
    )
    if global_pos_rate < 0.10 or global_pos_rate > 0.90:
        n_pos = int(y_meta.sum())
        n_neg = len(y_meta) - n_pos
        ideal_spw = n_neg / n_pos if n_pos > 0 else 1.0
        ideal_threshold = global_pos_rate  # threshold próximo da taxa base calibra o ponto de corte
        logger.warning(
            "Desbalanceamento severo de classes ({:.1f}% positivos). "
            "scale_pos_weight ideal ≈ {:.1f} (neg/pos = {}/{}). "
            "meta_threshold funcional ≈ {:.2f}–{:.2f} (baseado na taxa de positivos). "
            "Verifique também se max_holding da Triple Barrier é compatível com o horizonte do Alpha.",
            global_pos_rate * 100, ideal_spw, n_neg, n_pos,
            ideal_threshold * 0.5, ideal_threshold * 1.5,
        )
    avg_fold_size = len(y_meta) * (1 - 1 / 6)  # estimativa grosseira do fold de treino no CPCV 6/2
    hessian_per_sample = 0.25  # p*(1-p) com p≈0.5 no início
    min_samples_needed = min_child_weight / hessian_per_sample
    if min_samples_needed > avg_fold_size * 0.3:
        logger.warning(
            "min_child_weight={:.1f} exige ~{:.0f} amostras por nó — alto para folds de ~{:.0f} amostras. "
            "O XGBoost pode não conseguir criar splits (AUC→0.5). Considere reduzir min_child_weight.",
            min_child_weight, min_samples_needed, avg_fold_size,
        )

    for i, (train_idx, test_idx) in enumerate(splits):
        if len(train_idx) < 10: continue

        X_train, y_train = X.iloc[train_idx], y_meta.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y_meta.iloc[test_idx]

        # Diagnóstico de desbalanceamento — crucial para interpretar AUC
        pos_rate = y_train.mean()
        if pos_rate < 0.10 or pos_rate > 0.90:
            logger.warning(
                "Fold {}: desbalanceamento extremo no treino — {:.1f}% positivos ({} de {}). "
                "Considere revisar o Alpha ou os parâmetros da Triple Barrier.",
                i + 1, pos_rate * 100, int(y_train.sum()), len(y_train),
            )

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

        # Log AUC treino vs. teste por fold
        if len(np.unique(y_test)) > 1:
            import warnings

            from sklearn.metrics import roc_auc_score as _auc
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fold_train_auc = _auc(y_train, model.predict_proba(X_train)[:, 1])
                fold_test_auc = _auc(y_test, probs)
            gap = fold_train_auc - fold_test_auc
            if fold_train_auc <= 0.52 and fold_test_auc <= 0.52:
                # Modelo não aprendeu nada — underfitting (min_child_weight alto demais ou dados insuficientes)
                logger.warning(
                    "Fold {}: AUC treino={:.3f} | AUC teste={:.3f} | UNDERFITTING — modelo não discrimina. "
                    "Verifique min_child_weight vs. tamanho do fold e desbalanceamento de classes.",
                    i + 1, fold_train_auc, fold_test_auc,
                )
            elif gap > 0.15:
                logger.warning(
                    "Fold {}: AUC treino={:.3f} | AUC teste={:.3f} | gap={:.3f} — possível overfitting",
                    i + 1, fold_train_auc, fold_test_auc, gap,
                )
            else:
                logger.info(
                    "Fold {}: AUC treino={:.3f} | AUC teste={:.3f} | gap={:.3f}",
                    i + 1, fold_train_auc, fold_test_auc, gap,
                )

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

    # Calcula filter_rate: fração de sinais do Alpha que o Meta-Model rejeitou.
    # filter_rate = 1 - (trades_após_filtro / total_sinais_alpha)
    total_signals = len(y_test_all)
    accepted_signals = int(y_pred_all.sum())
    filter_rate = 1.0 - (accepted_signals / total_signals) if total_signals > 0 else 0.0
    logger.info(
        "Meta-Model filter rate: {:.3f} ({} sinais → {} aceitos)",
        filter_rate,
        total_signals,
        accepted_signals,
    )

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
        # Custo round-trip baseado em CostConfig: slippage (entrada + saída) + corretagem por contrato.
        # slippage_bps / 10_000 converte bps para fração; ×2 = round trip.
        # brokerage_per_contract em R$ dividido pelo preço em pontos (proxy de fração de contrato).
        "cost": (
            2 * cost_config.slippage_bps / 10_000
            + 2 * cost_config.brokerage_per_contract / df.loc[y_test_all.index, "close"]
        )
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

    # --- Minimum trade count guard (after ALL filters) ---
    n_trades_after_filters = len(trades)
    if n_trades_after_filters < 30:
        logger.warning(
            "Trade count after all filters is {} (< 30). "
            "Results will be statistically unreliable. "
            "Consider relaxing voi_threshold, Hurst threshold, or CUSUM sensitivity.",
            n_trades_after_filters,
        )

    return {
        "sharpe": attr_results["sharpe_full_system"],
        "sharpe_alpha": attr_results["sharpe_alpha_only"],
        "sharpe_lift": attr_results["sharpe_lift_meta"],
        "sharpe_train": avg_train_sharpe,
        "calmar_ratio": calmar,
        "n_trades": len(trades),
        "filter_rate": filter_rate,
        "vir_filter_rate": vir_filter_rate,
        "net_returns": net_returns,
        "alpha_returns": alpha_returns,
    }

def main():
    """Ponto de entrada de execução CLI do subsistema simulado do Backtest AFML.

    Inicia e configura os extratores locais dependentes da chamada do argparser.
    """
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

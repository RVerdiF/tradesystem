"""
Indicadores Técnicos e de Microestrutura — TradeSystem5000.

Este módulo implementa features financeiras point-in-time (sem look-ahead),
organizadas em categorias de momentum, volatilidade e microestrutura.

Categorias:
- **Momentum**: RSI, ROC, Distância de Médias Móveis.
- **Volatilidade**: ATR, Rolling Vol, Garman-Klass, Momentos (Skew/Kurtosis).
- **Microestrutura**: Order Flow Imbalance (OFI), VPIN, Volume Spread Analysis.

Funcionalidades:
- **compute_all_features**: Geração em lote de todas as features para o pipeline.
- Implementações vetorizadas via Pandas/NumPy para eficiência.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulos 18 e 19.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import feature_config


# ===========================================================================
# Momentum
# ===========================================================================
def moving_average_distance(close: pd.Series, period: int) -> pd.Series:
    """
    Calcula a distância percentual do preço para a média móvel selecionada.

    Normaliza o momentum e evita limites fixos (como no RSI), sendo mais
    informativo para modelos de ML sobre o "estiramento" do preço.

    Parameters
    ----------
    close : pd.Series
        Série de preços.
    period : int
        Período da EMA.

    Returns
    -------
    pd.Series
        Distância percentual: (Close - MA) / MA.
    """
    ma = close.ewm(span=period, adjust=False).mean()
    result = (close - ma) / ma.replace(0, np.nan)
    result.name = f"ma_dist_{period}"
    return result


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Rate of Change (retorno percentual em ``period`` barras).

    Parameters
    ----------
    close : pd.Series
        Série de preços.
    period : int
        Número de períodos.

    Returns
    -------
    pd.Series
        ROC em percentual.
    """
    result = close.pct_change(periods=period) * 100.0
    result.name = "roc"
    return result


# ===========================================================================
# Volatilidade
# ===========================================================================
def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int | None = None,
) -> pd.Series:
    """
    Average True Range.

    Parameters
    ----------
    high, low, close : pd.Series
        Séries OHLC.
    period : int, optional
        Período do ATR. Default: config.

    Returns
    -------
    pd.Series
        ATR (sempre positivo).
    """
    if period is None:
        period = feature_config.atr_period

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    result.name = "atr"
    return result


def rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Desvio padrão móvel dos retornos logarítmicos.

    Parameters
    ----------
    close : pd.Series
        Série de preços.
    window : int
        Janela do desvio padrão.

    Returns
    -------
    pd.Series
        Volatilidade realizada (log-retornos).
    """
    log_ret = np.log(close / close.shift(1))
    result = log_ret.rolling(window=window, min_periods=max(1, window // 4)).std()
    result.name = "rolling_vol"
    return result


def garman_klass_volatility(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Estimador de volatilidade Garman-Klass (baseado em OHLC).

    Incorpora a informação dos preços de abertura, máxima, mínima e fechamento,
    sendo mais eficiente que estimadores baseados apenas no fechamento.
    """
    log_hl = np.log(high / low.replace(0, np.nan)).pow(2)
    log_co = np.log(close / open_p.replace(0, np.nan)).pow(2)

    # gk_per_bar = 0.5 * (ln(H/L))^2 - (2ln2 - 1) * (ln(C/O))^2
    gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

    # Rolling mean para estabilização (variância -> desvio padrão)
    result = np.sqrt(gk_var.rolling(window=window, min_periods=window // 2).mean())
    result.name = "garman_klass"
    return result


def rolling_moments(close: pd.Series, window: int = 40) -> pd.DataFrame:
    """
    Calcula Skewness e Kurtosis móvel dos retornos.

    Ajuda a detectar exaustão de tendência e excesso de cauda (eventos extremos).
    """
    returns = close.pct_change()
    skew = returns.rolling(window=window, min_periods=window // 2).skew()
    kurt = returns.rolling(window=window, min_periods=window // 2).kurt()

    return pd.DataFrame({"skew": skew, "kurt": kurt}, index=close.index)


# ===========================================================================
# Regime Detection (Hurst Exponent)
# ===========================================================================
def rescaled_range_analysis(
    series: pd.Series,
    max_lag: int | None = None,
) -> float:
    """
    Calcula o Expoente de Hurst via análise R/S (Rescaled Range).

    O expoente de Hurst (H) classifica o comportamento da série:
    - H = 0.5: passeio aleatório (random walk)
    - H > 0.5: série persistente (tendência / momentum)
    - H < 0.5: série anti-persistente (reversão à média)

    Método: log(R/S) = H * log(n) + C, estimado via OLS sobre múltiplos lags.

    Parameters
    ----------
    series : pd.Series
        Série de preços ou retornos.
    max_lag : int, optional
        Lag máximo para a regressão. Default: len(series) // 2, capped at 60.

    Returns
    -------
    float
        Expoente de Hurst estimado. NaN se dados insuficientes.
    """
    if len(series) < 20:
        return np.nan

    if max_lag is None:
        max_lag = min(len(series) // 2, 60)

    # Usa retornos logarítmicos para estabilidade numérica
    log_ret = np.log(series / series.shift(1)).dropna()
    if len(log_ret) < max_lag:
        return np.nan

    # Calcula R/S para cada lag
    rs_values = []
    lag_values = []

    # Otimização: Extrai os valores para array numpy fora do loop
    # para evitar o overhead de slicing do Pandas .iloc em loop aninhado.
    log_ret_values = log_ret.values

    for lag in range(10, max_lag + 1, 5):  # step=5 para eficiência
        if lag > len(log_ret_values):
            break

        # Divide a série em sub-séries de tamanho 'lag'
        n_sub = len(log_ret_values) // lag
        if n_sub < 2:
            continue

        rs_sum = 0.0
        valid_chunks = 0

        for j in range(n_sub):
            chunk = log_ret_values[j * lag : (j + 1) * lag]
            mean_chunk = np.mean(chunk)
            cum_dev = np.cumsum(chunk - mean_chunk)

            r_range = np.max(cum_dev) - np.min(cum_dev)
            s_std = np.std(chunk, ddof=1)

            if s_std > 0 and r_range > 0:
                rs_sum += r_range / s_std
                valid_chunks += 1

        if valid_chunks > 0:
            avg_rs = rs_sum / valid_chunks
            if avg_rs > 0:
                rs_values.append(np.log(avg_rs))
                lag_values.append(np.log(lag))

    if len(rs_values) < 3:
        return np.nan

    # Regressão linear OLS: log(R/S) = H * log(n) + C
    x = np.array(lag_values)
    y = np.array(rs_values)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return np.nan

    hurst = numerator / denominator

    # Clampa a [0, 1] para validade matemática
    return float(np.clip(hurst, 0.0, 1.0))


def rolling_hurst_exponent(
    close: pd.Series,
    window: int = 100,
    step: int = 1,
) -> pd.Series:
    """
    Calcula o Expoente de Hurst em janela rolante.

    Parameters
    ----------
    close : pd.Series
        Série de preços de fechamento.
    window : int, optional
        Tamanho da janela rolante. Default: 100.
    step : int, optional
        Passo de cálculo (>1 para performance). Default: 1.

    Returns
    -------
    pd.Series
        Série com o Expoente de Hurst em cada timestamp.
    """
    result = pd.Series(np.nan, index=close.index, dtype=np.float64)
    result.name = "hurst_exponent"

    # Pré-computa a cada 'step' barras para eficiência
    indices = range(window, len(close), step)

    for i in indices:
        window_data = close.iloc[i - window : i]
        h = rescaled_range_analysis(window_data)
        if not np.isnan(h):
            result.iloc[i] = h

    # Forward-fill para preencher gaps entre cálculos (quando step > 1)
    result = result.ffill()

    return result


# ===========================================================================
# Microestrutura
# ===========================================================================
def volume_spread_analysis(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_p: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """
    Features de Volume Spread Analysis (VSA).

    Avalia a relação entre o esforço (volume) e o resultado (movimento de preço).
    """
    spread = high - low
    avg_spread = spread.rolling(window=20, min_periods=10).mean()
    rel_spread = spread / avg_spread.replace(0, np.nan)

    # Posição do fechamento na barra (0 = low, 1 = high)
    bar_pos = (close - low) / spread.replace(0, np.nan)

    # Volume relativo
    avg_vol = volume.rolling(window=20, min_periods=10).mean()
    rel_vol = volume / avg_vol.replace(0, np.nan)

    # Candle Wicks (Sombras)
    # +1 = sombra superior dominante, -1 = inferior dominante
    upper_wick = high - np.maximum(open_p, close)
    lower_wick = np.minimum(open_p, close) - low
    wick_ratio = (upper_wick - lower_wick) / spread.replace(0, np.nan)

    return pd.DataFrame(
        {
            "vsa_rel_spread": rel_spread,
            "vsa_bar_pos": bar_pos,
            "vsa_rel_vol": rel_vol,
            "vsa_wick_ratio": wick_ratio,
        },
        index=close.index,
    )


def order_flow_imbalance(volume: pd.Series, close: pd.Series) -> pd.Series:
    """
    Order Flow Imbalance estimado via tick rule.

    Classifica cada barra como compra (close > close anterior) ou venda,
    e calcula o desequilíbrio de volume.

    Parameters
    ----------
    volume : pd.Series
        Série de volume.
    close : pd.Series
        Série de preços de fechamento.

    Returns
    -------
    pd.Series
        OFI: positivo = pressão compradora, negativo = pressão vendedora.
    """
    # Tick rule: +1 se preço subiu, -1 se caiu, 0 se igual
    direction = np.sign(close.diff())
    # Forward fill zeros (mantém direção anterior)
    direction = direction.replace(0, np.nan).ffill().fillna(0)

    signed_volume = volume * direction

    # OFI como soma acumulada em janela
    ofi = signed_volume.rolling(window=20, min_periods=1).sum()
    ofi.name = "ofi"
    return ofi


def vpin(
    volume: pd.Series,
    close: pd.Series,
    n_buckets: int = 50,
) -> pd.Series:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN estimado).

    Estima a proporção de volume informado vs. total usando a classificação
    por tick rule e bucketing por volume.

    Parameters
    ----------
    volume : pd.Series
        Série de volume.
    close : pd.Series
        Série de preços.
    n_buckets : int
        Número de buckets para a estimativa rolling.

    Returns
    -------
    pd.Series
        VPIN em [0, 1]. Valores altos indicam mais informação assimétrica.
    """
    direction = np.sign(close.diff()).replace(0, np.nan).ffill().fillna(0)

    buy_volume = volume.where(direction > 0, 0.0)
    sell_volume = volume.where(direction < 0, 0.0)

    # VPIN = |buy - sell| / total em janela rolling
    buy_rolling = buy_volume.rolling(window=n_buckets, min_periods=max(1, n_buckets // 4)).sum()
    sell_rolling = sell_volume.rolling(window=n_buckets, min_periods=max(1, n_buckets // 4)).sum()
    total_rolling = volume.rolling(window=n_buckets, min_periods=max(1, n_buckets // 4)).sum()

    result = (buy_rolling - sell_rolling).abs() / total_rolling.replace(0, np.nan)
    result.name = "vpin"
    return result


def volume_imbalance(
    volume: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Volume Imbalance — pressão direcional de volume.

    Calcula o desequilíbrio entre volume comprador e vendedor usando tick rule,
    normalizado pelo volume total na janela. Valores positivos indicam pressão
    compradora dominante; negativos indicam pressão vendedora.

    Diferente do OFI (soma acumulada), este é normalizado para servir como
    gate de validação de sinais.

    Parameters
    ----------
    volume : pd.Series
        Série de volume (tick_volume ou real_volume).
    close : pd.Series
        Série de preços de fechamento.
    window : int, optional
        Janela para cálculo do desequilíbrio. Default: 20.

    Returns
    -------
    pd.Series
        Volume imbalance em [-1, 1]. 0 = equilíbrio.
    """
    # Tick rule: +1 se preço subiu, -1 se caiu
    direction = np.sign(close.diff())
    direction = direction.replace(0, np.nan).ffill().fillna(0)

    buy_volume = volume.where(direction > 0, 0.0)
    sell_volume = volume.where(direction < 0, 0.0)

    # Rolling sums
    buy_rolling = buy_volume.rolling(window=window, min_periods=max(1, window // 4)).sum()
    sell_rolling = sell_volume.rolling(window=window, min_periods=max(1, window // 4)).sum()
    total_rolling = volume.rolling(window=window, min_periods=max(1, window // 4)).sum()

    # Normalizado: (buy - sell) / total
    result = (buy_rolling - sell_rolling) / total_rolling.replace(0, np.nan)
    result.name = "volume_imbalance"
    return result


def volume_imbalance_zscore(
    volume: pd.Series,
    close: pd.Series,
    window: int = 20,
    z_window: int = 50,
) -> pd.Series:
    """
    Z-score do Volume Imbalance — detecta picos de pressão direcional.

    Útil para filtrar falsos rompimentos: um CUSUM trigger só é válido
    se houver um pico estatisticamente significativo no volume imbalance
    na mesma direção.

    Parameters
    ----------
    volume : pd.Series
        Série de volume.
    close : pd.Series
        Série de preços.
    window : int, optional
        Janela do volume imbalance. Default: 20.
    z_window : int, optional
        Janela para cálculo da média/desvio do imbalance. Default: 50.

    Returns
    -------
    pd.Series
        Z-score do volume imbalance. Valores > 0 = pressão compradora
        acima da média; valores < 0 = pressão vendedora abaixo da média.
    """
    imb = volume_imbalance(volume, close, window=window)

    rolling_mean = imb.rolling(window=z_window, min_periods=max(1, z_window // 4)).mean()
    rolling_std = imb.rolling(window=z_window, min_periods=max(1, z_window // 4)).std()

    zscore = (imb - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore.name = "volume_imbalance_zscore"
    return zscore


# ===========================================================================
# Geração em lote
# ===========================================================================
def compute_all_features(
    df: pd.DataFrame,
    config: object | None = None,
) -> pd.DataFrame:
    """
    Calcula todas as features a partir de um DataFrame OHLCV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas: ``open``, ``high``, ``low``, ``close``, ``volume``.
    config : FeatureConfig, optional
        Configuração. Default: config global.

    Returns
    -------
    pd.DataFrame
        DataFrame com todas as features calculadas.
    """
    if config is None:
        config = feature_config

    logger.info("Calculando todas as features ({} barras)", len(df))

    features = pd.DataFrame(index=df.index)

    # Momentum (Distância de Médias)
    features["ma_dist_fast"] = moving_average_distance(df["close"], period=config.ma_dist_fast_period)
    features["ma_dist_slow"] = moving_average_distance(df["close"], period=config.ma_dist_slow_period)
    features["roc"] = roc(df["close"])

    # Volatilidade
    features["atr"] = atr(df["high"], df["low"], df["close"], period=config.atr_period)
    features["rolling_vol"] = rolling_volatility(df["close"])
    features["garman_klass"] = garman_klass_volatility(
        df["open"], df["high"], df["low"], df["close"]
    )

    # Momentos (Skew/Kurt)
    moments_df = rolling_moments(df["close"], window=config.moments_window)
    features = pd.concat([features, moments_df], axis=1)

    # Microestrutura
    if "volume" in df.columns:
        features["ofi"] = order_flow_imbalance(df["volume"], df["close"])
        features["vpin"] = vpin(df["volume"], df["close"])
        features["volume_imbalance"] = volume_imbalance(df["volume"], df["close"])
        features["volume_imbalance_zscore"] = volume_imbalance_zscore(df["volume"], df["close"])

        vsa_df = volume_spread_analysis(df["high"], df["low"], df["close"], df["open"], df["volume"])
        features = pd.concat([features, vsa_df], axis=1)

    # Regime Detection (Hurst Exponent)
    features["hurst_exponent"] = rolling_hurst_exponent(
        df["close"],
        window=getattr(config, "hurst_window", 100),
        step=getattr(config, "hurst_step", 5),
    )

    logger.success("Features calculadas: {} colunas", len(features.columns))
    return features

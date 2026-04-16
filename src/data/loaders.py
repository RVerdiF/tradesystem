import sys
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta

from src.data.mt5_connector import mt5_session
from src.data.extractor import INTERVAL_TO_TF, extract_ohlc


def fetch_yfinance_data(
    symbol: str = "PETR4.SA", years: float = 5, interval: str = "1d"
) -> pd.DataFrame:
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
            logger.warning(
                f"Intervalo {interval} limitado a {max_d} dias no Yahoo Finance. "
                f"Usando {max_d} dias."
            )
    else:
        days = requested_days

    start_date = datetime.now() - timedelta(days=days)

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, interval=interval)

    if df.empty:
        logger.error(f"Nenhum dado retornado do yfinance para {symbol}.")
        sys.exit(1)

    # Padroniza as colunas (todas minúsculas)
    df.columns = [c.lower() for c in df.columns]

    # No yfinance, timezone tz-aware pode causar problema em algumas funções rolling.
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    logger.success(f"Dados baixados: {len(df)} barras ({interval}).")
    return df


def fetch_mt5_data(
    symbol: str = "PETR4", years: float = 5, interval: str = "1h", n_bars: int = 5000
) -> pd.DataFrame:
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
        with mt5_session():
            df = extract_ohlc(symbol=symbol, timeframe=tf, n_bars=n_bars)

        if df.empty:
            logger.error(f"Nenhum dado encontrado no MT5 para {symbol}.")
            sys.exit(1)

        # O extractor já retorna time como index e colunas em minúsculas
        # Usa tick_volume como volume (comum no backtest MT5)
        df = (
            df[["open", "high", "low", "close", "tick_volume"]]
            .rename(columns={"tick_volume": "volume"})
            .dropna()
        )
        logger.success(f"Dados baixados: {len(df)} barras do MT5.")
        return df

    except Exception as e:
        logger.error(f"Falha ao conectar no MT5: {e}")
        logger.info("Sugestão: garanta que o Terminal MT5 está aberto e com AutoTrading ativo.")
        sys.exit(1)

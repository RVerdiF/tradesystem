"""
Módulo Alpha (Sinal Direcional Primário) — TradeSystem5000.

Este módulo define a interface e implementações para o modelo de Alpha,
responsável por gerar a aposta direcional primária do sistema {-1, 0, +1}.

Implementações:
- **TrendFollowingAlpha**: Estratégia de seguimento de tendência via cruzamento de EMAs.
- **MeanReversionAlpha**: Estratégia de reversão à média baseada em Z-score dinâmico.

O sinal do Alpha serve como gatilho (trigger) para o método da Tripla Barreira.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulo 3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import feature_config, labeling_config


# ---------------------------------------------------------------------------
# Hurst Exponent Utility
# ---------------------------------------------------------------------------
def compute_hurst_exponent(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Compute the rolling Hurst Exponent using the Rescaled Range (R/S) method.

    H > 0.5  =>  trending / persistent series
    H == 0.5 =>  random walk
    H < 0.5  =>  mean-reverting series

    Parameters
    ----------
    series : pd.Series
        Price or indicator series. Index must be a DatetimeIndex.
        For best results, pass close_fracdiff (stationary) rather than raw close.
    window : int
        Rolling window length. Minimum recommended: 100 bars.
        Values below 60 are mathematically unreliable.

    Returns
    -------
    pd.Series
        Rolling Hurst exponent, aligned to the input index.
        NaN for the first (window - 1) observations and wherever std == 0.
    """
    if window < 60:
        raise ValueError(
            f"compute_hurst_exponent: window={window} is below the minimum of 60. "
            "Hurst estimates on short windows are statistically unreliable. "
            "Use window >= 100 for production."
        )

    def _rs_hurst(arr: np.ndarray) -> float:
        """Single-window R/S Hurst calculation. Called by rolling().apply()."""
        n = len(arr)
        mean_adj = arr - arr.mean()
        cumdev = np.cumsum(mean_adj)
        r = cumdev.max() - cumdev.min()
        s = arr.std(ddof=1)
        if s == 0.0 or r == 0.0:
            return np.nan
        return np.log(r / s) / np.log(n / 2)

    return series.rolling(window=window).apply(_rs_hurst, raw=True)


# ---------------------------------------------------------------------------
# Interface base
# ---------------------------------------------------------------------------
class AlphaModel(ABC):
    """Interface abstrata para modelos geradores de sinal direcional."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Gera sinais de trading a partir dos dados.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame OHLCV (ou features).

        Returns
        -------
        pd.Series
            Série com valores em {-1, 0, +1}.
            +1 = compra, -1 = venda, 0 = neutro.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome identificador do modelo Alpha."""
        ...


# ---------------------------------------------------------------------------
# Trend Following — Cruzamento de EMAs
# ---------------------------------------------------------------------------
class TrendFollowingAlpha(AlphaModel):
    """
    Alpha baseado em seguimento de tendência (cruzamento de EMAs).

    Sinal (Modo Normal):
    - +1 quando EMA rápida > EMA lenta (tendência de alta)
    - -1 quando EMA rápida < EMA lenta (tendência de baixa)

    Sinal (Modo Reversão):
    - -1 quando EMA rápida > EMA lenta
    - +1 quando EMA rápida < EMA lenta

    Parameters
    ----------
    fast_span : int, optional
        Span da EMA rápida. Default: config.
    slow_span : int, optional
        Span da EMA lenta. Default: config.
    reversion_mode : bool, optional
        Se True, inverte o sinal (opera contra a tendência). Default: True (conforme Plano de Reversão).
    """

    def __init__(
        self,
        fast_span: int | None = None,
        slow_span: int | None = None,
        reversion_mode: bool = True,
    ) -> None:
        self.fast_span = fast_span or labeling_config.trend_fast_span
        self.slow_span = slow_span or labeling_config.trend_slow_span
        self.reversion_mode = reversion_mode

    @property
    def name(self) -> str:
        mode_str = "Reversion" if self.reversion_mode else "Trend"
        return f"TrendFollowing[{mode_str}](fast={self.fast_span}, slow={self.slow_span})"

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Gera sinais de cruzamento de EMA sobre a série de preço real (não-estacionária).

        Parameters
        ----------
        df : pd.DataFrame
            Deve conter coluna ``close`` (preço real de fechamento OHLCV).
            Para o filtro opcional de Hurst, ``close_fracdiff`` é preferido quando disponível.

        Returns
        -------
        pd.Series
            +1 (compra), -1 (venda), ou 0 (neutro, durante warm-up ou filtros).

        Notes
        -----
        Arquitetura AFML (López de Prado, 2018, Cap. 3):
        - O **sinal primário (Alpha)** baseado em regras técnicas é uma heurística
          do trader e DEPENDE da não-estacionaridade para capturar drift/tendência.
          Por isso, o cruzamento de EMAs é computado sobre ``close`` (preço real).
        - O **meta-modelo** e demais features operam sobre ``close_fracdiff``
          (estacionária, com memória preservada via FFD).
        - A rotulação via Triple Barrier é a ponte entre os dois mundos.
        """
        if "close" not in df.columns:
            raise KeyError(
                "TrendFollowingAlpha.generate_signal: coluna 'close' ausente. "
                "O Alpha opera sobre a série de preço real (não-estacionária)."
            )
        price_series = df["close"]

        ema_fast = price_series.ewm(span=self.fast_span, adjust=False).mean()
        ema_slow = price_series.ewm(span=self.slow_span, adjust=False).mean()

        signal = pd.Series(0, index=price_series.index, dtype=np.int8, name="signal")

        # Lógica base: +1 se fast > slow (alta), -1 se fast < slow (baixa)
        if self.reversion_mode:
            signal[ema_fast < ema_slow] = 1
            signal[ema_fast > ema_slow] = -1
        else:
            signal[ema_fast > ema_slow] = 1
            signal[ema_fast < ema_slow] = -1

        signal[ema_fast == ema_slow] = 0

        # Warm-up: marca NaN até que ambas as EMAs estejam estáveis
        warmup = max(self.fast_span, self.slow_span)
        signal.iloc[:warmup] = 0

        return signal


# ---------------------------------------------------------------------------
# Composite Alpha
# ---------------------------------------------------------------------------
class CompositeAlpha(AlphaModel):
    """
    Composite Alpha combining Trend Following (EMAs), Hurst Exponent regime filter,
    and Volume Imbalance Z-Score filter.

    Signal generation requires simultaneous agreement:
    1. EMA fast > EMA slow (for long) or EMA fast < EMA slow (for short).
    2. Hurst Exponent > hurst_threshold (default 0.55 indicating trending regime).
    3. Volume Imbalance Z-Score (volume_imbalance_zscore) > vir_zscore_threshold (for long) or < -vir_zscore_threshold (for short).

    Parameters
    ----------
    long_fast_span : int, optional
        Fast EMA span for long signals. Default: config.
    long_slow_span : int, optional
        Slow EMA span for long signals. Default: config.
    short_fast_span : int, optional
        Fast EMA span for short signals. Default: config.
    short_slow_span : int, optional
        Slow EMA span for short signals. Default: config.
    long_hurst_threshold : float, optional
        Minimum Hurst Exponent required for long. Default: config.
    short_hurst_threshold : float, optional
        Minimum Hurst Exponent required for short. Default: config.
    long_vir_zscore_threshold : float, optional
        Minimum Volume Imbalance Z-Score required for long. Default: config.
    short_vir_zscore_threshold : float, optional
        Minimum Volume Imbalance Z-Score required for short. Default: config.
    """

    def __init__(
        self,
        long_fast_span: int | None = None,
        long_slow_span: int | None = None,
        short_fast_span: int | None = None,
        short_slow_span: int | None = None,
        long_hurst_threshold: float | None = None,
        short_hurst_threshold: float | None = None,
        long_vir_zscore_threshold: float | None = None,
        short_vir_zscore_threshold: float | None = None,
    ) -> None:
        self.long_fast_span = long_fast_span or labeling_config.long_fast_span
        self.long_slow_span = long_slow_span or labeling_config.long_slow_span
        self.short_fast_span = short_fast_span or labeling_config.short_fast_span
        self.short_slow_span = short_slow_span or labeling_config.short_slow_span
        self.long_hurst_threshold = long_hurst_threshold or feature_config.long_hurst_threshold
        self.short_hurst_threshold = short_hurst_threshold or feature_config.short_hurst_threshold
        self.long_vir_zscore_threshold = (
            long_vir_zscore_threshold if long_vir_zscore_threshold is not None else feature_config.long_vol_imbalance_z_threshold
        )
        self.short_vir_zscore_threshold = (
            short_vir_zscore_threshold if short_vir_zscore_threshold is not None else feature_config.short_vol_imbalance_z_threshold
        )

    @property
    def name(self) -> str:
        return (
            f"CompositeAlpha("
            f"L:[f={self.long_fast_span}, s={self.long_slow_span}, h>{self.long_hurst_threshold}, vz>{self.long_vir_zscore_threshold}], "
            f"S:[f={self.short_fast_span}, s={self.short_slow_span}, h>{self.short_hurst_threshold}, vz<-{self.short_vir_zscore_threshold}]"
            f")"
        )

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on the decoupled composite rules.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'close', 'hurst_exponent', and 'volume_imbalance_zscore' columns.

        Returns
        -------
        pd.Series
            Series of {-1, 0, +1} signals.
        """
        required_cols = ["close", "hurst_exponent", "volume_imbalance_zscore"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"CompositeAlpha.generate_signal: column '{col}' missing.")

        price_series = df["close"]

        long_ema_fast = price_series.ewm(span=self.long_fast_span, adjust=False).mean()
        long_ema_slow = price_series.ewm(span=self.long_slow_span, adjust=False).mean()

        short_ema_fast = price_series.ewm(span=self.short_fast_span, adjust=False).mean()
        short_ema_slow = price_series.ewm(span=self.short_slow_span, adjust=False).mean()

        signal = pd.Series(0, index=price_series.index, dtype=np.int8, name="signal")

        # Long conditions
        long_cond = (
            (long_ema_fast > long_ema_slow)
            & (df["hurst_exponent"] > self.long_hurst_threshold)
            & (df["volume_imbalance_zscore"] > self.long_vir_zscore_threshold)
        )

        # Short conditions
        short_cond = (
            (short_ema_fast < short_ema_slow)
            & (df["hurst_exponent"] > self.short_hurst_threshold)
            & (df["volume_imbalance_zscore"] < -self.short_vir_zscore_threshold)
        )

        signal[long_cond] = 1
        signal[short_cond] = -1

        # Warm-up: set to NaN until EMAs are stable for both sides
        warmup = max(self.long_fast_span, self.long_slow_span, self.short_fast_span, self.short_slow_span)
        signal.iloc[:warmup] = 0

        return signal


# ---------------------------------------------------------------------------
# Mean Reversion — Z-score
# ---------------------------------------------------------------------------
class MeanReversionAlpha(AlphaModel):
    """
    Alpha baseado em reversão à média via Z-score, aplicado sobre série fracionária estacionária.

    Sinal:
    - +1 (compra) quando Z-score < -entry (preço muito abaixo da média)
    - -1 (venda) quando Z-score > +entry (preço muito acima da média)
    - 0 (neutro) quando |Z-score| < entry

    O sinal permanece ativo até que Z-score cruze ``exit_threshold``.

    Parameters
    ----------
    window : int, optional
        Janela da média/desvio. Default: config.
    entry_threshold : float, optional
        Z-score mínimo para abrir posição. Default: config.
    exit_threshold : float, optional
        Z-score para fechar posição. Default: config.
    """

    def __init__(
        self,
        window: int | None = None,
        entry_threshold: float | None = None,
        exit_threshold: float | None = None,
    ) -> None:
        self.window = window or labeling_config.mean_rev_window
        self.entry_threshold = (
            entry_threshold if entry_threshold is not None else labeling_config.mean_rev_entry
        )
        self.exit_threshold = (
            exit_threshold if exit_threshold is not None else labeling_config.mean_rev_exit
        )

    @property
    def name(self) -> str:
        return (
            f"MeanReversion(w={self.window}, "
            f"entry={self.entry_threshold}, exit={self.exit_threshold})"
        )

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Gera sinais de reversão à média (Z-score) sobre a série fracionariamente diferenciada.

        Parameters
        ----------
        df : pd.DataFrame
            Deve conter coluna ``close_fracdiff`` (série estacionária).

        Returns
        -------
        pd.Series
            +1 (compra), -1 (venda), ou 0 (neutro).

        Notes
        -----
        De acordo com AFML, técnicas estatísticas como o Z-score pressupõem estacionariedade.
        Operar sobre séries brutas de preço introduz heterocedasticidade. Usamos `close_fracdiff`
        para manter a validade matemática do limite de reversão.
        """
        if "close_fracdiff" not in df.columns:
            raise KeyError(
                "MeanReversionAlpha.generate_signal: coluna 'close_fracdiff' ausente. "
                "O Z-Score deve ser aplicado sobre série estacionária."
            )
        price_series = df["close_fracdiff"]

        rolling_mean = price_series.rolling(window=self.window, min_periods=self.window).mean()
        rolling_std = price_series.rolling(window=self.window, min_periods=self.window).std()
        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (price_series - rolling_mean) / rolling_std

        signal = pd.Series(0, index=price_series.index, dtype=np.int8, name="signal")

        # Gera sinais com hysteresis (entry/exit)
        position = 0
        for i in range(len(zscore)):
            z = zscore.iloc[i]

            if np.isnan(z):
                signal.iloc[i] = 0
                continue

            if position == 0:
                # Sem posição: entra se Z-score suficiente
                if z < -self.entry_threshold:
                    position = 1   # compra (preço abaixo da média)
                elif z > self.entry_threshold:
                    position = -1  # venda (preço acima da média)
            elif position == 1:
                # Comprado: sai quando Z-score chega ao exit
                if z >= self.exit_threshold:
                    position = 0
            elif position == -1:
                # Vendido: sai quando Z-score chega ao -exit
                if z <= -self.exit_threshold:
                    position = 0

            signal.iloc[i] = position

        n_long = (signal == 1).sum()
        n_short = (signal == -1).sum()
        logger.info(
            "{} — sinais: {} long, {} short, {} neutro",
            self.name,
            n_long,
            n_short,
            (signal == 0).sum(),
        )
        return signal


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def get_signal_events(signal: pd.Series) -> pd.DatetimeIndex:
    """
    Extrai timestamps onde o sinal muda (transições de posição).

    Útil para definir os momentos de amostragem na Tripla Barreira.

    Parameters
    ----------
    signal : pd.Series
        Série de sinais {-1, 0, +1}.

    Returns
    -------
    pd.DatetimeIndex
        Timestamps onde houve mudança de sinal (excluindo transição para 0).
    """
    changes = signal.diff().abs()
    # Filtra apenas mudanças que resultam em posição ativa (!= 0)
    events = signal[(changes > 0) & (signal != 0)]
    logger.debug("Eventos de sinal: {} mudanças detectadas", len(events))
    return pd.DatetimeIndex(events.index)

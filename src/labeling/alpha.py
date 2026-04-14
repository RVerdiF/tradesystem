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
# Filter utilities
# ---------------------------------------------------------------------------
def apply_regime_filter(
    signal: pd.Series,
    hurst: pd.Series,
    threshold: float = 0.55,
) -> pd.Series:
    """
    Aplica filtro de regime de mercado via Expoente de Hurst.

    O sinal do Alpha só é validado se H > threshold (tendência persistente).
    Se H <= threshold (passeio aleatório ou reversão), o sinal é zerado.

    Parameters
    ----------
    signal : pd.Series
        Sinal original do Alpha {-1, 0, +1}.
    hurst : pd.Series
        Expoente de Hurst em cada timestamp.
    threshold : float, optional
        Hurst mínimo para validar tendência. Default: 0.55.

    Returns
    -------
    pd.Series
        Sinal filtrado por regime.
    """
    filtered = signal.copy()

    # Zera sinais onde Hurst indica não-tendência
    # H <= threshold: random walk ou mean-reverting → side = 0
    no_trend_mask = (hurst <= threshold) | hurst.isna()
    filtered[no_trend_mask] = 0

    n_zeroed = ((signal != 0) & no_trend_mask).sum()
    logger.info(
        "Regime filter (Hurst <= {}): {} sinais zerados de {}",
        threshold,
        n_zeroed,
        (signal != 0).sum(),
    )

    return filtered


def apply_volume_imbalance_filter(
    signal: pd.Series,
    vol_imb_zscore: pd.Series,
    z_threshold: float = 0.5,
) -> pd.Series:
    """
    Aplica filtro de Volume Imbalance para validar sinais.

    O sinal do Alpha só passa se houver pressão de volume compatível:
    - side = +1 requer volume_imbalance_zscore > +z_threshold
    - side = -1 requer volume_imbalance_zscore < -z_threshold

    Parameters
    ----------
    signal : pd.Series
        Sinal original do Alpha {-1, 0, +1}.
    vol_imb_zscore : pd.Series
        Z-score do volume imbalance em cada timestamp.
    z_threshold : float, optional
        Z-score mínimo para validar pressão de volume. Default: 0.5.

    Returns
    -------
    pd.Series
        Sinal filtrado por volume imbalance.
    """
    filtered = signal.copy()

    # Para sinais long (+1): requer zscore > threshold
    long_mask = (signal == 1) & (vol_imb_zscore <= z_threshold)
    filtered[long_mask] = 0

    # Para sinais short (-1): requer zscore < -threshold
    short_mask = (signal == -1) & (vol_imb_zscore >= -z_threshold)
    filtered[short_mask] = 0

    n_long_zeroed = ((signal == 1) & long_mask).sum()
    n_short_zeroed = ((signal == -1) & short_mask).sum()
    logger.info(
        "Volume filter (|z| > {}): {} long e {} short zerados",
        z_threshold,
        n_long_zeroed,
        n_short_zeroed,
    )

    return filtered


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

    Filtros opcionais:
    - regime_filter: filtra sinais via Hurst Exponent (H > 0.55 = tendência)
    - volume_filter: filtra sinais via Volume Imbalance Z-Score

    Parameters
    ----------
    fast_span : int, optional
        Span da EMA rápida. Default: config.
    slow_span : int, optional
        Span da EMA lenta. Default: config.
    reversion_mode : bool, optional
        Se True, inverte o sinal (opera contra a tendência). Default: True (conforme Plano de Reversão).
    enable_regime_filter : bool, optional
        Se True, aplica filtro de regime via Hurst Exponent. Default: False.
    enable_volume_filter : bool, optional
        Se True, aplica filtro de Volume Imbalance. Default: False.
    hurst_threshold : float, optional
        Threshold do Hurst para validar tendência. Default: config.
    vol_imbalance_z_threshold : float, optional
        Z-score mínimo do volume imbalance. Default: config.
    """

    def __init__(
        self,
        fast_span: int | None = None,
        slow_span: int | None = None,
        reversion_mode: bool = True,
        enable_regime_filter: bool = False,
        enable_volume_filter: bool = False,
        hurst_threshold: float | None = None,
        vol_imbalance_z_threshold: float | None = None,
    ) -> None:
        self.fast_span = fast_span or labeling_config.trend_fast_span
        self.slow_span = slow_span or labeling_config.trend_slow_span
        self.reversion_mode = reversion_mode
        self.enable_regime_filter = enable_regime_filter
        self.enable_volume_filter = enable_volume_filter
        self.hurst_threshold = hurst_threshold if hurst_threshold is not None else feature_config.hurst_threshold
        self.vol_imbalance_z_threshold = vol_imbalance_z_threshold if vol_imbalance_z_threshold is not None else feature_config.vol_imbalance_z_threshold

    @property
    def name(self) -> str:
        mode_str = "Reversion" if self.reversion_mode else "Trend"
        return f"TrendFollowing[{mode_str}](fast={self.fast_span}, slow={self.slow_span})"

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Gera sinais de cruzamento de EMA sobre série estacionária.

        Parameters
        ----------
        df : pd.DataFrame
            Deve conter coluna ``close_fracdiff`` (Diferenciação Fracionária).
            Se ausente, faz fallback para ``close`` com aviso.

        Returns
        -------
        pd.Series
            +1 (compra) ou -1 (venda). Sem zeros — sempre posicionado.

        Notes
        -----
        López de Prado (2018): operar sobre preços brutos (não-estacionários)
        gera regressões espúrias. A série FracDiff é estacionária e orbita
        em torno de um valor médio constante (geralmente zero).
        """
        if "close_fracdiff" in df.columns:
            price_series = df["close_fracdiff"]
        else:
            logger.warning(
                "TrendFollowingAlpha: coluna 'close_fracdiff' ausente. "
                "Fallback para 'close' (série não-estacionária — risco de sinais espúrios)."
            )
            price_series = df["close"]

        ema_fast = price_series.ewm(span=self.fast_span, adjust=False).mean()
        ema_slow = price_series.ewm(span=self.slow_span, adjust=False).mean()

        signal = pd.Series(0, index=price_series.index, dtype=np.int8, name="signal")

        # Lógica base: +1 se fast > slow (alta), -1 se fast < slow (baixa)
        if not self.reversion_mode:
            signal[ema_fast > ema_slow] = 1
            signal[ema_fast < ema_slow] = -1
        else:
            # Modo Reversão: -1 se fast > slow, +1 se fast < slow
            signal[ema_fast > ema_slow] = -1
            signal[ema_fast < ema_slow] = 1

        signal[ema_fast == ema_slow] = 0

        # Warm-up: marca NaN até que ambas as EMAs estejam estáveis
        warmup = max(self.fast_span, self.slow_span)
        signal.iloc[:warmup] = 0

        # Aplica filtro de regime (Hurst Exponent) se habilitado
        if self.enable_regime_filter and "hurst_exponent" in df.columns:
            signal = apply_regime_filter(
                signal,
                df["hurst_exponent"],
                threshold=self.hurst_threshold,
            )

        # Aplica filtro de volume imbalance se habilitado
        if self.enable_volume_filter and "volume_imbalance_zscore" in df.columns:
            signal = apply_volume_imbalance_filter(
                signal,
                df["volume_imbalance_zscore"],
                z_threshold=self.vol_imbalance_z_threshold,
            )

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
# Mean Reversion — Z-score
# ---------------------------------------------------------------------------
class MeanReversionAlpha(AlphaModel):
    """
    Alpha baseado em reversão à média via Z-score.

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
        Gera sinais de reversão à média sobre série estacionária.

        Parameters
        ----------
        df : pd.DataFrame
            Deve conter coluna ``close_fracdiff`` (Diferenciação Fracionária).
            Se ausente, faz fallback para ``close`` com aviso.

        Returns
        -------
        pd.Series
            +1 (compra), -1 (venda), ou 0 (neutro).

        Notes
        -----
        A série FracDiff é estacionária e orbita em torno de zero.
        O Z-score detecta desvios extremos da média estacionária.
        """
        if "close_fracdiff" in df.columns:
            price_series = df["close_fracdiff"]
        else:
            logger.warning(
                "MeanReversionAlpha: coluna 'close_fracdiff' ausente. "
                "Fallback para 'close' (série não-estacionária — risco de sinais espúrios)."
            )
            price_series = df["close"]

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

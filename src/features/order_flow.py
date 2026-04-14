"""
src/features/order_flow.py

Volume Imbalance Ratio (VIR) utilities.

NOTE: This is NOT VPIN (Volume-Synchronized Probability of Informed Trading).
VPIN (Easley, López de Prado & O'Hara) requires ultra-high-frequency data
bucketed by fixed volume intervals, not fixed-time OHLCV bars.

What we compute here is a Volume Imbalance Ratio (VIR) using the Tick Rule
(Lee & Ready, 1991) applied to OHLCV bars. This is a heuristic approximation.
On B3 mini-futures via MT5 retail, tick_volume counts price updates (not
contracts), so this indicator carries significant noise from HFT activity.
Validate all results against real tick data before production use.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def tick_rule_direction(close_series: pd.Series) -> pd.Series:
    """
    Apply the Tick Rule (Lee & Ready, 1991) to classify each bar as
    buyer-aggressor (+1) or seller-aggressor (-1).

    Rules:
    - close[t] > close[t-1]  → uptick  → buy aggression  (+1)
    - close[t] < close[t-1]  → downtick → sell aggression (-1)
    - close[t] == close[t-1] → zero-tick → propagate previous direction (ffill)

    The first bar has no prior close; it is set to 0 (no information).

    Parameters
    ----------
    close_series : pd.Series
        Series of close prices, indexed by timestamp.

    Returns
    -------
    pd.Series
        Series of {-1, 0, +1} with same index as close_series.
        0 only appears at the very first bar (no prior close available).
    """
    diff = close_series.diff()

    # Assign raw direction: +1 for uptick, -1 for downtick, NaN for zero-tick
    direction = pd.Series(np.nan, index=close_series.index, dtype=float)
    direction[diff > 0] = 1.0
    direction[diff < 0] = -1.0
    # diff == 0 stays NaN → will be forward-filled (zero-tick rule)

    # Forward-fill zero-tick bars with the last known direction
    direction = direction.ffill()

    # The very first bar has no prior close: ffill produces NaN → set to 0
    direction = direction.fillna(0.0)

    return direction


def compute_vir(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute the Volume Imbalance Ratio (VIR).

    VIR[t] = rolling_sum(buy_vol - sell_vol, window) / rolling_sum(total_vol, window)

    Where:
    - buy_vol[t]  = direction[t] * volume[t]  if direction[t] > 0, else 0
    - sell_vol[t] = |direction[t] * volume[t]| if direction[t] < 0, else 0
    - total_vol[t] = volume[t]

    VIR is bounded in [-1, +1].
    - VIR near +1 → strong buyer aggression over the window
    - VIR near -1 → strong seller aggression over the window
    - VIR near  0 → balanced flow

    MT5 WARNING: If df["real_volume"] is all zeros, we fall back to
    df["tick_volume"]. Emits a warning log so the caller is aware.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: "close" and either "real_volume" or "volume"
        (tick_volume aliased as "volume" by the MT5 extractor).
    window : int
        Rolling window in bars for the VIR computation. Default 20.

    Returns
    -------
    pd.Series
        VIR series aligned to df.index. NaN where total volume is zero.
    """
    # --- Volume column selection ---
    if "real_volume" in df.columns and df["real_volume"].sum() > 0:
        vol = df["real_volume"].copy()
    elif "volume" in df.columns:
        vol = df["volume"].copy()
        if "real_volume" in df.columns:
            logger.warning(
                "compute_vir: real_volume is all zeros. "
                "Falling back to tick_volume (df['volume']). "
                "VIR computed from tick counts, NOT contract volume. "
                "Results are approximate; validate against real tick data."
            )
    else:
        raise KeyError("compute_vir: df must contain column 'volume' or 'real_volume'.")

    # --- Tick Rule direction ---
    direction = tick_rule_direction(df["close"])

    # --- Directional volume ---
    # buy_vol > 0 only where direction == +1; sell_vol > 0 only where direction == -1
    signed_vol = direction * vol
    buy_vol = signed_vol.clip(lower=0)  # keeps positive values, zeros negative
    sell_vol = (-signed_vol).clip(lower=0)  # absolute value of negative signed_vol

    # --- Rolling sums ---
    rolling_buy = buy_vol.rolling(window=window, min_periods=1).sum()
    rolling_sell = sell_vol.rolling(window=window, min_periods=1).sum()
    rolling_total = vol.rolling(window=window, min_periods=1).sum()

    # --- VIR: normalised imbalance ---
    # Where total volume is zero (e.g., auction bars), return NaN
    vir = (rolling_buy - rolling_sell) / rolling_total.replace(0, np.nan)

    vir.name = "vir"
    return vir


def compute_vir_zscore(vir_series: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute the rolling z-score of the VIR series.

    z = (VIR[t] - mean(VIR[t-window:t])) / std(VIR[t-window:t])

    This converts VIR (bounded [-1, +1]) into a standardised score that
    allows threshold-based filtering regardless of the asset's typical
    imbalance magnitude.

    'Peak in direction' is defined as vir_zscore > voi_threshold (configurable).
    Default threshold in Optuna search space: 1.0 (one std above rolling mean).

    Parameters
    ----------
    vir_series : pd.Series
        Output of compute_vir().
    window : int
        Rolling window for mean and std. Default 20.

    Returns
    -------
    pd.Series
        Rolling z-score of VIR. NaN where std is zero or VIR is NaN.
        No inf values (zero std replaced with NaN).
    """
    rolling_mean = vir_series.rolling(window=window, min_periods=1).mean()
    rolling_std = vir_series.rolling(window=window, min_periods=1).std()

    # Replace zero std with NaN to avoid inf values
    rolling_std = rolling_std.replace(0, np.nan)

    zscore = (vir_series - rolling_mean) / rolling_std
    zscore.name = "vir_zscore"

    # Defensive check: log if any inf leaked through
    n_inf = np.isinf(zscore).sum()
    if n_inf > 0:
        logger.warning(
            "compute_vir_zscore: %d inf values detected in vir_zscore. "
            "Replacing with NaN.",
            n_inf,
        )
        zscore = zscore.replace([np.inf, -np.inf], np.nan)

    return zscore

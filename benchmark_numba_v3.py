import time
import numpy as np
import pandas as pd
from numba import njit, float64, int64

@njit
def fast_rs_analysis(series_values, max_lag):
    n = len(series_values)
    if n < 20:
        return np.nan

    log_ret = np.zeros(n - 1)
    for i in range(1, n):
        if series_values[i-1] != 0:
            log_ret[i-1] = np.log(series_values[i] / series_values[i-1])
        else:
            log_ret[i-1] = np.nan

    # Remove nans equivalent
    valid_idx = np.where(~np.isnan(log_ret))[0]
    log_ret = log_ret[valid_idx]

    if len(log_ret) < max_lag:
        return np.nan

    # Using arrays instead of lists to avoid Numba list overhead
    rs_values = np.zeros(max_lag)
    lag_values = np.zeros(max_lag)
    count = 0

    for lag in range(10, max_lag + 1, 5):
        if lag > len(log_ret):
            break

        n_sub = len(log_ret) // lag
        if n_sub < 2:
            continue

        rs_sum = 0.0
        valid_chunks = 0

        for j in range(n_sub):
            chunk = log_ret[j * lag : (j + 1) * lag]
            mean_chunk = np.mean(chunk)

            # cumsum equivalent
            cum_dev = np.zeros(len(chunk))
            curr_sum = 0.0
            for k in range(len(chunk)):
                curr_sum += chunk[k] - mean_chunk
                cum_dev[k] = curr_sum

            r_range = np.max(cum_dev) - np.min(cum_dev)

            # fast std with ddof=1
            chunk_var = 0.0
            for k in range(len(chunk)):
                chunk_var += (chunk[k] - mean_chunk) ** 2
            s_std = np.sqrt(chunk_var / (len(chunk) - 1)) if len(chunk) > 1 else 0.0

            if s_std > 0 and r_range > 0:
                rs_sum += r_range / s_std
                valid_chunks += 1

        if valid_chunks > 0:
            avg_rs = rs_sum / valid_chunks
            if avg_rs > 0:
                rs_values[count] = np.log(avg_rs)
                lag_values[count] = np.log(lag)
                count += 1

    if count < 3:
        return np.nan

    x = lag_values[:count]
    y = rs_values[:count]

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return np.nan

    hurst = numerator / denominator

    # clip
    if hurst < 0.0:
        return 0.0
    elif hurst > 1.0:
        return 1.0
    return float(hurst)

@njit
def fast_rolling_hurst_numba(values, window, step, max_lag):
    n = len(values)
    result = np.full(n, np.nan)

    for i in range(window, n, step):
        window_data = values[i - window : i]
        h = fast_rs_analysis(window_data, max_lag)
        result[i] = h

    return result

def fast_rolling_hurst(close, window=100, step=1):
    result = pd.Series(np.nan, index=close.index, dtype=np.float64)
    values = close.values

    max_lag = min(window // 2, 60)
    out = fast_rolling_hurst_numba(values, window, step, max_lag)

    result = pd.Series(out, index=close.index)
    return result.ffill()

np.random.seed(42)
close = pd.Series(np.cumsum(np.random.randn(10000)) + 1000) # prevent negative values for log

start = time.time()
res = fast_rolling_hurst(close, window=100, step=1)
end = time.time()

print(f"Time: {end - start:.4f} seconds")

# Run pure python equivalent one more time to compare to Numba without compilation
start = time.time()
res_fast = fast_rolling_hurst(close, window=100, step=1)
end = time.time()

print(f"Time (without compilation): {end - start:.4f} seconds")

# Ensure they output same results
np.random.seed(42)
close_small = pd.Series(np.cumsum(np.random.randn(200)) + 100)
from src.features.indicators import rolling_hurst_exponent
r1 = rolling_hurst_exponent(close_small, window=100, step=1)
r2 = fast_rolling_hurst(close_small, window=100, step=1)

diff = np.abs(r1 - r2).max()
print(f"Max diff: {diff}")

import time
import numpy as np
import pandas as pd
from src.features.indicators import rolling_hurst_exponent

np.random.seed(42)
close = pd.Series(np.cumsum(np.random.randn(10000)))

start = time.time()
res = rolling_hurst_exponent(close, window=100, step=1)
end = time.time()

print(f"Time: {end - start:.4f} seconds")

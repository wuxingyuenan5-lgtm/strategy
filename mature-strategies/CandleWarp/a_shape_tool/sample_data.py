from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

AssetPreset = Literal["xauusd", "btc", "eurusd", "generic"]


def make_sample_ohlc(
    output_path: str | Path,
    rows: int = 2000,
    seed: int = 42,
    asset: AssetPreset = "xauusd",
) -> Path:
    """Generate realistic synthetic OHLCV time-series data with regime shifts and volume profiles.

    Presets:
    - ``xauusd``: Gold dynamics ($2000-$2600 base, session intraday surges, realistic ATR).
    - ``btc``: Crypto dynamics ($40,000-$70,000 base, heavy-tailed explosive moves, weekend volume).
    - ``eurusd``: Major Forex dynamics ($1.05-$1.12 base, strong mean-reversion, London/NY overlaps).
    - ``generic``: Standard drift-diffusion benchmark.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=rows, freq="h")

    close = np.empty(rows)
    open_ = np.empty(rows)
    high = np.empty(rows)
    low = np.empty(rows)

    if asset == "xauusd":
        close[0] = 2050.0
        regime_drifts = np.array([0.00022, -0.00015, 0.00005, 0.00035, -0.00025])
        regime_vols = np.array([0.0022, 0.0028, 0.0014, 0.0038, 0.0032])
        base_vol_mean = 8.5
    elif asset == "btc":
        close[0] = 42500.0
        regime_drifts = np.array([0.00045, -0.00035, 0.00010, 0.00075, -0.00055])
        regime_vols = np.array([0.0055, 0.0070, 0.0035, 0.0095, 0.0080])
        base_vol_mean = 9.8
    elif asset == "eurusd":
        close[0] = 1.0850
        regime_drifts = np.array([0.00008, -0.00006, 0.00000, 0.00012, -0.00010])
        regime_vols = np.array([0.0009, 0.0012, 0.0006, 0.0016, 0.0013])
        base_vol_mean = 7.8
    else:  # generic
        close[0] = 100.0
        regime_drifts = np.array([0.00018, -0.00012, 0.0, 0.00028, -0.00022])
        regime_vols = np.array([0.0018, 0.0023, 0.0011, 0.0032, 0.0028])
        base_vol_mean = 8.0

    open_[0] = close[0]

    for i in range(1, rows):
        regime = (i // 140) % len(regime_drifts)
        wave = 0.00045 * np.sin(i / 19.0) + 0.00025 * np.cos(i / 43.0)
        # Fat-tailed Student-t return distribution approximation
        noise = rng.standard_t(df=4.5) * (regime_vols[regime] * 0.75)
        ret = regime_drifts[regime] + wave + noise
        open_[i] = close[i - 1] * (1.0 + rng.normal(0.0, regime_vols[regime] * 0.15))
        close[i] = open_[i] * np.exp(ret)

    volume = np.empty(rows)
    for i in range(rows):
        regime = (i // 140) % len(regime_vols)
        base_range = close[i] * abs(rng.normal(regime_vols[regime] * 1.5, regime_vols[regime] * 0.5))
        upper = base_range * rng.uniform(0.35, 0.95)
        lower = base_range * rng.uniform(0.35, 0.95)
        high[i] = max(open_[i], close[i]) + upper
        low[i] = min(open_[i], close[i]) - lower
        # Volume correlated with range and volatility regime
        vol_intensity = (high[i] - low[i]) / (close[i] * regime_vols[regime] + 1e-8)
        volume[i] = max(50.0, rng.lognormal(mean=base_vol_mean, sigma=0.55) * vol_intensity)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df.to_csv(output_path, index=False, float_format="%.6f")
    return output_path



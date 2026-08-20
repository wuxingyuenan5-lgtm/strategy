"""pivot.py — ZigZag-based swing-point (pivot) detection.

A pivot high is a local peak where the move up from the prior pivot low is at
least ``atr_mult * ATR``.  A pivot low is the mirror.  This is more robust than
a fixed-window rolling-max approach because it adapts to current volatility and
avoids counting tiny noise wiggles as pivots during low-volatility periods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Direction = Literal["high", "low"]


@dataclass(frozen=True)
class Pivot:
    index: int          # bar index in the original DataFrame
    price: float        # close price at the pivot
    direction: Direction  # "high" or "low"
    timestamp: object   # timestamp value (for display)


def find_pivots(
    df: pd.DataFrame,
    atr_mult: float = 0.5,
    atr_col: str = "atr",
) -> list[Pivot]:
    """Return a list of Pivot objects detected via a ZigZag algorithm.

    Parameters
    ----------
    df:
        DataFrame with at least ``high``, ``low``, ``close``, and ``atr``
        columns and a ``timestamp`` column.
    atr_mult:
        Minimum move size (in ATR units) required to register a new pivot.
        Larger values → fewer, more significant pivots.
    atr_col:
        Name of the ATR column in *df*.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = df[atr_col].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy()

    pivots: list[Pivot] = []

    # Seed: find first bar with a valid ATR
    start = int(np.argmax(np.isfinite(atr)))
    if not np.isfinite(atr[start]):
        return pivots

    # ZigZag state
    last_dir: Direction = "low"   # direction of the last confirmed pivot
    last_price = low[start]
    last_idx = start
    candidate_price = last_price
    candidate_idx = start

    for i in range(start + 1, len(df)):
        threshold = atr_mult * atr[i] if np.isfinite(atr[i]) else 0.0

        if last_dir == "low":
            # Looking for the next high
            if high[i] > candidate_price:
                candidate_price = high[i]
                candidate_idx = i
            # Check if price has reversed enough from candidate high on a SUBSEQUENT bar
            if i > candidate_idx and candidate_price - low[i] >= threshold:
                # Confirm the candidate high as a pivot
                pivots.append(Pivot(
                    index=candidate_idx,
                    price=candidate_price,
                    direction="high",
                    timestamp=timestamps[candidate_idx],
                ))
                last_dir = "high"
                last_price = candidate_price
                last_idx = candidate_idx
                candidate_price = low[i]
                candidate_idx = i
        else:
            # last_dir == "high" — looking for the next low
            if low[i] < candidate_price:
                candidate_price = low[i]
                candidate_idx = i
            # Check if price has reversed enough from candidate low on a SUBSEQUENT bar
            if i > candidate_idx and high[i] - candidate_price >= threshold:
                # Confirm the candidate low as a pivot
                pivots.append(Pivot(
                    index=candidate_idx,
                    price=candidate_price,
                    direction="low",
                    timestamp=timestamps[candidate_idx],
                ))
                last_dir = "low"
                last_price = candidate_price
                last_idx = candidate_idx
                candidate_price = high[i]
                candidate_idx = i

    return pivots


def pivots_to_frame(pivots: list[Pivot]) -> pd.DataFrame:
    """Convert a list of Pivot objects to a tidy DataFrame."""
    if not pivots:
        return pd.DataFrame(columns=["index", "price", "direction", "timestamp"])
    return pd.DataFrame([
        {"index": p.index, "price": p.price, "direction": p.direction, "timestamp": p.timestamp}
        for p in pivots
    ])

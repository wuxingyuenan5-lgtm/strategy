"""vp_fvg.py — Volume Profile (VP) and Fair Value Gap (FVG) microstructure feature engineering."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolumeProfileMetrics:
    poc_rel: float
    vah_rel: float
    val_rel: float
    va_width_rel: float
    vp_skew: float
    vol_mean_rel: float


@dataclass(frozen=True)
class FVGMetrics:
    bull_fvg_count: int
    bear_fvg_count: int
    net_fvg_bias: float
    nearest_fvg_dist: float
    is_retesting_fvg: bool


def compute_volume_profile(
    window: pd.DataFrame,
    atr: float,
    n_bins: int = 20,
    value_area_pct: float = 0.70,
) -> VolumeProfileMetrics:
    """Compute local Volume Profile (VP) metrics over a window slice.

    Extracts:
    - POC (Point of Control): Price bin with the highest trading volume.
    - VAH / VAL (Value Area High / Low): The 70% volume containment area.
    - VP Skew: Volume asymmetry above vs below POC (-1 to +1).
    - Normalized volume intensity.
    """
    if len(window) == 0:
        return VolumeProfileMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    first_close = float(window["close"].iloc[0])
    atr_safe = max(float(atr), 1e-8)

    # If volume is missing or all zeros, generate synthetic tick intensity from range
    if "volume" not in window.columns or (window["volume"] <= 0).all():
        vol = (window["high"] - window["low"]).to_numpy(dtype=float) + 1e-6
    else:
        vol = window["volume"].to_numpy(dtype=float)
        # replace any zero/negative with small epsilon
        vol = np.where(vol <= 0, np.nanmedian(vol[vol > 0]) if np.any(vol > 0) else 1.0, vol)

    lows = window["low"].to_numpy(dtype=float)
    highs = window["high"].to_numpy(dtype=float)

    min_p = float(np.min(lows))
    max_p = float(np.max(highs))

    if max_p - min_p < 1e-8:
        return VolumeProfileMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    bin_edges = np.linspace(min_p, max_p, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_volumes = np.zeros(n_bins, dtype=np.float64)

    # Distribute bar volume across bins intersected by [low_i, high_i]
    for l_val, h_val, v_val in zip(lows, highs, vol):
        bar_range = max(h_val - l_val, 1e-8)
        # Find overlapping bins
        overlap_mask = (bin_edges[1:] >= l_val) & (bin_edges[:-1] <= h_val)
        if not np.any(overlap_mask):
            idx = np.clip(np.searchsorted(bin_edges, 0.5 * (l_val + h_val)) - 1, 0, n_bins - 1)
            bin_volumes[idx] += v_val
        else:
            # Overlap proportion
            overlap_starts = np.maximum(bin_edges[:-1], l_val)
            overlap_ends = np.minimum(bin_edges[1:], h_val)
            weights = np.maximum(0.0, overlap_ends - overlap_starts) / bar_range
            bin_volumes += v_val * weights

    total_vol = float(np.sum(bin_volumes))
    if total_vol <= 0:
        total_vol = 1.0

    poc_idx = int(np.argmax(bin_volumes))
    poc_price = float(bin_centers[poc_idx])

    # Compute Value Area (70% around POC)
    target_vol = total_vol * value_area_pct
    current_vol = bin_volumes[poc_idx]
    left_idx, right_idx = poc_idx, poc_idx

    while current_vol < target_vol and (left_idx > 0 or right_idx < n_bins - 1):
        next_left = bin_volumes[left_idx - 1] if left_idx > 0 else -1.0
        next_right = bin_volumes[right_idx + 1] if right_idx < n_bins - 1 else -1.0

        if next_left >= next_right and left_idx > 0:
            left_idx -= 1
            current_vol += bin_volumes[left_idx]
        elif right_idx < n_bins - 1:
            right_idx += 1
            current_vol += bin_volumes[right_idx]
        else:
            break

    val_price = float(bin_edges[left_idx])
    vah_price = float(bin_edges[right_idx + 1])

    # Volume skewness: (Vol above POC - Vol below POC) / total_vol
    vol_above = float(np.sum(bin_volumes[poc_idx + 1:])) if poc_idx < n_bins - 1 else 0.0
    vol_below = float(np.sum(bin_volumes[:poc_idx])) if poc_idx > 0 else 0.0
    vp_skew = (vol_above - vol_below) / total_vol

    # Dimensionless scaling
    poc_rel = (poc_price - first_close) / atr_safe
    vah_rel = (vah_price - first_close) / atr_safe
    val_rel = (val_price - first_close) / atr_safe
    va_width_rel = (vah_price - val_price) / atr_safe

    vol_mean_rel = float(np.mean(vol) / (np.median(vol) + 1e-8))

    return VolumeProfileMetrics(
        poc_rel=float(poc_rel),
        vah_rel=float(vah_rel),
        val_rel=float(val_rel),
        va_width_rel=float(va_width_rel),
        vp_skew=float(np.clip(vp_skew, -1.0, 1.0)),
        vol_mean_rel=float(vol_mean_rel),
    )


def detect_unmitigated_fvg(
    window: pd.DataFrame,
    atr: float,
    min_gap_atr: float = 0.15,
) -> FVGMetrics:
    """Detect Fair Value Gaps (3-bar price imbalances) and track their unmitigated state.

    - Bullish FVG: Low[i] > High[i-2] (gap interval = [High[i-2], Low[i]])
    - Bearish FVG: High[i] < Low[i-2] (gap interval = [High[i], Low[i-2]])
    - Tracks whether subsequent candles in the window tested or filled the gap.
    """
    n = len(window)
    if n < 3:
        return FVGMetrics(0, 0, 0.0, 0.0, False)

    first_close = float(window["close"].iloc[0])
    last_close = float(window["close"].iloc[-1])
    atr_safe = max(float(atr), 1e-8)

    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)

    unmitigated_bull: list[tuple[float, float, float]] = []  # (gap_low, gap_high, mid)
    unmitigated_bear: list[tuple[float, float, float]] = []

    for i in range(2, n):
        # 1. Check for Bullish FVG formed at bar i
        if lows[i] > highs[i - 2]:
            gap_low = highs[i - 2]
            gap_high = lows[i]
            gap_size = gap_high - gap_low
            if gap_size >= min_gap_atr * atr_safe:
                # Check if mitigated by future bars in [i+1, n-1]
                mitigated = False
                for j in range(i + 1, n):
                    if lows[j] <= gap_low:
                        mitigated = True
                        break
                if not mitigated:
                    unmitigated_bull.append((gap_low, gap_high, 0.5 * (gap_low + gap_high)))

        # 2. Check for Bearish FVG formed at bar i
        elif highs[i] < lows[i - 2]:
            gap_low = highs[i]
            gap_high = lows[i - 2]
            gap_size = gap_high - gap_low
            if gap_size >= min_gap_atr * atr_safe:
                mitigated = False
                for j in range(i + 1, n):
                    if highs[j] >= gap_high:
                        mitigated = True
                        break
                if not mitigated:
                    unmitigated_bear.append((gap_low, gap_high, 0.5 * (gap_low + gap_high)))

    bull_count = len(unmitigated_bull)
    bear_count = len(unmitigated_bear)

    # Net bias in ATR units
    bull_energy = sum((gh - gl) / atr_safe for gl, gh, _ in unmitigated_bull)
    bear_energy = sum((gh - gl) / atr_safe for gl, gh, _ in unmitigated_bear)
    net_fvg_bias = float(bull_energy - bear_energy)

    # Find nearest active FVG to current close
    all_active = unmitigated_bull + unmitigated_bear
    if all_active:
        mids = np.array([m for _, _, m in all_active])
        dists = mids - last_close
        nearest_idx = int(np.argmin(np.abs(dists)))
        nearest_dist_atr = float(dists[nearest_idx] / atr_safe)

        # Check if currently retesting the nearest gap (within 0.5 ATR)
        is_retest = bool(abs(nearest_dist_atr) < 0.5)
    else:
        nearest_dist_atr = 0.0
        is_retest = False

    return FVGMetrics(
        bull_fvg_count=bull_count,
        bear_fvg_count=bear_count,
        net_fvg_bias=net_fvg_bias,
        nearest_fvg_dist=nearest_dist_atr,
        is_retesting_fvg=is_retest,
    )

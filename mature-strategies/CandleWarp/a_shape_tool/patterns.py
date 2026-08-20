"""patterns.py — Fixed-structure chart-pattern scanner.

Supported patterns
------------------
* ``w_bottom``          — Double Bottom (W)
* ``m_top``             — Double Top (M)
* ``head_shoulders``    — Head & Shoulders (bearish)
* ``inv_head_shoulders``— Inverse Head & Shoulders (bullish)

Each pattern is located in pivot space.  After an optional breakout-confirmation
step the scanner records the ``horizon`` bars that follow and computes per-path
and quantile statistics identical to the existing similarity pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .pivot import Pivot, find_pivots

PatternName = Literal[
    "w_bottom", "m_top", "head_shoulders", "inv_head_shoulders", "converging_triangle",
    "diverging_triangle", "falling_wedge", "rising_wedge"
]



# ---------------------------------------------------------------------------
# Config & result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatternConfig:
    pattern: PatternName = "w_bottom"
    horizon: int = 50
    # Pivot detection
    pivot_atr_mult: float = 0.5
    # Time constraints (in bars) between the two anchor pivots
    min_gap: int = 20
    max_gap: int = 200
    # Price symmetry tolerance (in ATR units):
    #   |price(anchor1) - price(anchor2)| < price_tolerance * ATR
    price_tolerance: float = 0.5
    # Minimum height of the pattern body (in ATR units):
    #   neckline - trough (for W) must be >= min_height * ATR
    min_height: float = 0.5
    # For H&S: maximum deviation of shoulders from each other (ATR units)
    shoulder_tolerance: float = 0.8
    # Whether to require a confirmed breakout before recording the pattern
    confirm_breakout: bool = True
    # Maximum bars to wait for breakout confirmation after pattern forms
    breakout_window: int = 20
    timeframe: str = "1h"
    # Precision tuning fields
    preceding_trend_window: int = 40
    preceding_trend_strength: float = 1.5
    min_leg_spacing: int = 10


@dataclass
class PatternMatch:
    pattern: PatternName
    # Key pivot indices (into the original DataFrame)
    pivots: list[int]          # e.g. [L1, H_neck, L2] for W-bottom
    pivot_prices: list[float]
    pivot_times: list[object]
    neckline: float            # breakout level
    confirm_index: int         # bar where breakout is confirmed (or last pivot)
    confirm_time: object
    forward_path: np.ndarray   # length horizon+1, normalised return from confirm bar
    terminal_return: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forward_path(close: np.ndarray, start: int, horizon: int) -> np.ndarray | None:
    """Return a horizon+1 return path starting at *start*, or None if too short."""
    end = start + horizon
    if end >= len(close):
        return None
    segment = close[start: end + 1].astype(float)
    ref = segment[0]
    if ref <= 0 or not np.isfinite(ref):
        return None
    return segment / ref - 1.0


def _atr_at(atr: np.ndarray, idx: int) -> float:
    """Return ATR at *idx*, falling back to the last finite value."""
    val = atr[idx]
    if np.isfinite(val) and val > 0:
        return val
    finite = atr[:idx + 1][np.isfinite(atr[:idx + 1])]
    return float(finite[-1]) if len(finite) else 1.0


def _check_preceding_trend(
    close: np.ndarray,
    idx: int,
    atr_val: float,
    trend_type: Literal["up", "down"],
    window: int = 40,
    min_atr_change: float = 1.5,
) -> bool:
    """Check if the price prior to *idx* was weakening (down) or strengthening (up)."""
    if idx < window:
        return False
    preceding = close[idx - window : idx]
    if len(preceding) < window:
        return False
        
    # Fit a simple linear regression: y = slope * x + intercept
    x = np.arange(window)
    slope, intercept = np.polyfit(x, preceding, 1)
    
    # Total change based on the trendline
    total_change = slope * window
    
    if trend_type == "down":
        # Consistently weakening:
        is_decreasing = slope < 0
        significant_drop = total_change <= -min_atr_change * atr_val
        # True bottom validation: close[idx] must be in the lower part of the trend period (relaxed to 1.2 ATR)
        is_true_bottom = close[idx] <= np.min(preceding) + 1.2 * atr_val
        
        return is_decreasing and significant_drop and is_true_bottom
    else: # trend_type == "up"
        # Consistently strengthening:
        is_increasing = slope > 0
        significant_rise = total_change >= min_atr_change * atr_val
        # True peak validation: close[idx] must be in the upper part of the trend period (relaxed to 1.2 ATR)
        is_true_peak = close[idx] >= np.max(preceding) - 1.2 * atr_val
        
        return is_increasing and significant_rise and is_true_peak



# ---------------------------------------------------------------------------
# Pattern scanners
# ---------------------------------------------------------------------------

def _scan_w_bottom(
    df: pd.DataFrame,
    pivots: list[Pivot],
    config: PatternConfig,
    close: np.ndarray,
    atr: np.ndarray,
) -> list[PatternMatch]:
    """Double-bottom: Low → High(neck) → Low → (breakout above neck)."""
    matches: list[PatternMatch] = []
    lows = [p for p in pivots if p.direction == "low"]

    for i in range(len(lows) - 1):
        l1, l2 = lows[i], lows[i + 1]
        gap = l2.index - l1.index
        if not (config.min_gap <= gap <= config.max_gap):
            continue

        atr_val = _atr_at(atr, l2.index)

        # 1. Preceding trend check: rigorous consistent weakening before first bottom
        # Use fixed preceding window to capture the immediate trend
        trend_win = config.preceding_trend_window
        if not _check_preceding_trend(
            close, l1.index, atr_val, "down", window=trend_win, min_atr_change=config.preceding_trend_strength
        ):
            continue

        # Find highest pivot between the two lows → neckline
        neck_candidates = [
            p for p in pivots
            if p.direction == "high" and l1.index < p.index < l2.index
        ]
        if not neck_candidates:
            continue
        neck = max(neck_candidates, key=lambda p: p.price)

        # 3. Structural separation check: ensure space between bottoms
        if neck.index - l1.index < config.min_leg_spacing or l2.index - neck.index < config.min_leg_spacing:
            continue

        # Symmetry check
        if abs(l1.price - l2.price) > config.price_tolerance * atr_val:
            continue
        # Height check
        if neck.price - max(l1.price, l2.price) < config.min_height * atr_val:
            continue

        # 4. Determine confirmation bar with invalidation support
        if config.confirm_breakout:
            confirm_idx = None
            search_end = min(l2.index + config.breakout_window + 1, len(close))
            support_level = min(l1.price, l2.price) - 0.2 * atr_val
            for bar in range(l2.index + 1, search_end):
                # If price drops below support, W-bottom is invalidated
                if close[bar] < support_level:
                    break
                if close[bar] > neck.price:
                    confirm_idx = bar
                    break
            if confirm_idx is None:
                continue
        else:
            confirm_idx = l2.index

        path = _forward_path(close, confirm_idx, config.horizon)
        if path is None:
            continue

        matches.append(PatternMatch(
            pattern="w_bottom",
            pivots=[l1.index, neck.index, l2.index],
            pivot_prices=[l1.price, neck.price, l2.price],
            pivot_times=[l1.timestamp, neck.timestamp, l2.timestamp],
            neckline=neck.price,
            confirm_index=confirm_idx,
            confirm_time=df["timestamp"].iloc[confirm_idx],
            forward_path=path,
            terminal_return=float(path[-1]),
        ))

    return matches


def _scan_m_top(
    df: pd.DataFrame,
    pivots: list[Pivot],
    config: PatternConfig,
    close: np.ndarray,
    atr: np.ndarray,
) -> list[PatternMatch]:
    """Double-top: High → Low(neck) → High → (breakout below neck)."""
    matches: list[PatternMatch] = []
    highs = [p for p in pivots if p.direction == "high"]

    for i in range(len(highs) - 1):
        h1, h2 = highs[i], highs[i + 1]
        gap = h2.index - h1.index
        if not (config.min_gap <= gap <= config.max_gap):
            continue

        atr_val = _atr_at(atr, h2.index)

        # 1. Preceding trend check: rigorous consistent strengthening before first top
        # Use fixed preceding window to capture the immediate trend
        trend_win = config.preceding_trend_window
        if not _check_preceding_trend(
            close, h1.index, atr_val, "up", window=trend_win, min_atr_change=config.preceding_trend_strength
        ):
            continue

        # Find lowest pivot between the two highs → neckline
        neck_candidates = [
            p for p in pivots
            if p.direction == "low" and h1.index < p.index < h2.index
        ]
        if not neck_candidates:
            continue
        neck = min(neck_candidates, key=lambda p: p.price)

        # 3. Structural separation check: ensure space between tops
        if neck.index - h1.index < config.min_leg_spacing or h2.index - neck.index < config.min_leg_spacing:
            continue

        # Symmetry check
        if abs(h1.price - h2.price) > config.price_tolerance * atr_val:
            continue
        # Height check
        if min(h1.price, h2.price) - neck.price < config.min_height * atr_val:
            continue

        # 4. Determine confirmation bar with invalidation support
        if config.confirm_breakout:
            confirm_idx = None
            search_end = min(h2.index + config.breakout_window + 1, len(close))
            resistance_level = max(h1.price, h2.price) + 0.2 * atr_val
            for bar in range(h2.index + 1, search_end):
                # If price rises above resistance, M-top is invalidated
                if close[bar] > resistance_level:
                    break
                if close[bar] < neck.price:
                    confirm_idx = bar
                    break
            if confirm_idx is None:
                continue
        else:
            confirm_idx = h2.index

        path = _forward_path(close, confirm_idx, config.horizon)
        if path is None:
            continue

        matches.append(PatternMatch(
            pattern="m_top",
            pivots=[h1.index, neck.index, h2.index],
            pivot_prices=[h1.price, neck.price, h2.price],
            pivot_times=[h1.timestamp, neck.timestamp, h2.timestamp],
            neckline=neck.price,
            confirm_index=confirm_idx,
            confirm_time=df["timestamp"].iloc[confirm_idx],
            forward_path=path,
            terminal_return=float(path[-1]),
        ))

    return matches


def _scan_head_shoulders(
    df: pd.DataFrame,
    pivots: list[Pivot],
    config: PatternConfig,
    close: np.ndarray,
    atr: np.ndarray,
) -> list[PatternMatch]:
    """Head & Shoulders: LS_high → V_left → Head_high → V_right → RS_high → breakout."""
    matches: list[PatternMatch] = []
    highs = [p for p in pivots if p.direction == "high"]
    lows = [p for p in pivots if p.direction == "low"]

    for i in range(len(highs) - 2):
        ls, head, rs = highs[i], highs[i + 1], highs[i + 2]

        total_gap = rs.index - ls.index
        if not (config.min_gap <= total_gap <= config.max_gap * 2):
            continue

        # Head must be highest
        if not (head.price > ls.price and head.price > rs.price):
            continue

        atr_val = _atr_at(atr, head.index)

        # 1. Preceding trend check: rigorous consistent strengthening before left shoulder
        # Use fixed preceding window to capture the immediate trend
        trend_win = config.preceding_trend_window
        if not _check_preceding_trend(
            close, ls.index, atr_val, "up", window=trend_win, min_atr_change=config.preceding_trend_strength
        ):
            continue

        # Shoulders roughly symmetric
        if abs(ls.price - rs.price) > config.shoulder_tolerance * atr_val:
            continue

        # Minimum head prominence
        if head.price - max(ls.price, rs.price) < config.min_height * atr_val:
            continue

        # Find neckline: lowest pivot in each valley
        left_valley = [p for p in lows if ls.index < p.index < head.index]
        right_valley = [p for p in lows if head.index < p.index < rs.index]
        if not left_valley or not right_valley:
            continue
        lv = min(left_valley, key=lambda p: p.price)
        rv = min(right_valley, key=lambda p: p.price)
        neckline = (lv.price + rv.price) / 2.0

        # 2. Structural separation check: ensure clear shoulders and head spacing
        leg_space = config.min_leg_spacing
        if (lv.index - ls.index < leg_space or head.index - lv.index < leg_space or
            rv.index - head.index < leg_space or rs.index - rv.index < leg_space):
            continue

        # 3. Determine confirmation bar with invalidation support (cannot break above head)
        if config.confirm_breakout:
            confirm_idx = None
            search_end = min(rs.index + config.breakout_window + 1, len(close))
            resistance_level = head.price + 0.1 * atr_val
            for bar in range(rs.index + 1, search_end):
                # Invalidated if price shoots above head
                if close[bar] > resistance_level:
                    break
                if close[bar] < neckline:
                    confirm_idx = bar
                    break
            if confirm_idx is None:
                continue
        else:
            confirm_idx = rs.index

        path = _forward_path(close, confirm_idx, config.horizon)
        if path is None:
            continue

        matches.append(PatternMatch(
            pattern="head_shoulders",
            pivots=[ls.index, lv.index, head.index, rv.index, rs.index],
            pivot_prices=[ls.price, lv.price, head.price, rv.price, rs.price],
            pivot_times=[ls.timestamp, lv.timestamp, head.timestamp, rv.timestamp, rs.timestamp],
            neckline=neckline,
            confirm_index=confirm_idx,
            confirm_time=df["timestamp"].iloc[confirm_idx],
            forward_path=path,
            terminal_return=float(path[-1]),
        ))

    return matches


def _scan_inv_head_shoulders(
    df: pd.DataFrame,
    pivots: list[Pivot],
    config: PatternConfig,
    close: np.ndarray,
    atr: np.ndarray,
) -> list[PatternMatch]:
    """Inverse Head & Shoulders: LS_low → H_left → Head_low → H_right → RS_low → breakout."""
    matches: list[PatternMatch] = []
    lows = [p for p in pivots if p.direction == "low"]
    highs = [p for p in pivots if p.direction == "high"]

    for i in range(len(lows) - 2):
        ls, head, rs = lows[i], lows[i + 1], lows[i + 2]

        total_gap = rs.index - ls.index
        if not (config.min_gap <= total_gap <= config.max_gap * 2):
            continue

        if not (head.price < ls.price and head.price < rs.price):
            continue

        atr_val = _atr_at(atr, head.index)

        # 1. Preceding trend check: rigorous consistent weakening before left shoulder
        # Use fixed preceding window to capture the immediate trend
        trend_win = config.preceding_trend_window
        if not _check_preceding_trend(
            close, ls.index, atr_val, "down", window=trend_win, min_atr_change=config.preceding_trend_strength
        ):
            continue

        if abs(ls.price - rs.price) > config.shoulder_tolerance * atr_val:
            continue
        if min(ls.price, rs.price) - head.price < config.min_height * atr_val:
            continue

        left_peak = [p for p in highs if ls.index < p.index < head.index]
        right_peak = [p for p in highs if head.index < p.index < rs.index]
        if not left_peak or not right_peak:
            continue
        lp = max(left_peak, key=lambda p: p.price)
        rp = max(right_peak, key=lambda p: p.price)
        neckline = (lp.price + rp.price) / 2.0

        # 2. Structural separation check: ensure clear shoulders and head spacing
        leg_space = config.min_leg_spacing
        if (lp.index - ls.index < leg_space or head.index - lp.index < leg_space or
            rp.index - head.index < leg_space or rs.index - rp.index < leg_space):
            continue

        # 3. Determine confirmation bar with invalidation support (cannot break below head)
        if config.confirm_breakout:
            confirm_idx = None
            search_end = min(rs.index + config.breakout_window + 1, len(close))
            support_level = head.price - 0.1 * atr_val
            for bar in range(rs.index + 1, search_end):
                # Invalidated if price drops below head
                if close[bar] < support_level:
                    break
                if close[bar] > neckline:
                    confirm_idx = bar
                    break
            if confirm_idx is None:
                continue
        else:
            confirm_idx = rs.index

        path = _forward_path(close, confirm_idx, config.horizon)
        if path is None:
            continue

        matches.append(PatternMatch(
            pattern="inv_head_shoulders",
            pivots=[ls.index, lp.index, head.index, rp.index, rs.index],
            pivot_prices=[ls.price, lp.price, head.price, rp.price, rs.price],
            pivot_times=[ls.timestamp, lp.timestamp, head.timestamp, rp.timestamp, rs.timestamp],
            neckline=neckline,
            confirm_index=confirm_idx,
            confirm_time=df["timestamp"].iloc[confirm_idx],
            forward_path=path,
            terminal_return=float(path[-1]),
        ))

    return matches


def _scan_converging_triangle(
    df: pd.DataFrame,
    pivots: list[Pivot],
    config: PatternConfig,
    close: np.ndarray,
    atr: np.ndarray,
) -> list[PatternMatch]:
    """Converging Triangle (收敛三角形): 4 alternating pivots with contracting range.
    Highs descending/flat (H2 <= H1 + 0.15*ATR) and lows ascending/flat (L2 >= L1 - 0.15*ATR),
    with overall contracting width: (H2 - L2) < (H1 - L1) - 0.15 * ATR.
    """
    matches: list[PatternMatch] = []
    if len(pivots) < 4:
        return matches

    for i in range(len(pivots) - 3):
        p0, p1, p2, p3 = pivots[i], pivots[i + 1], pivots[i + 2], pivots[i + 3]
        dirs = [p0.direction, p1.direction, p2.direction, p3.direction]
        if dirs not in [["high", "low", "high", "low"], ["low", "high", "low", "high"]]:
            continue

        total_gap = p3.index - p0.index
        if not (config.min_gap <= total_gap <= config.max_gap * 1.5):
            continue

        if p0.direction == "high":
            h1, l1, h2, l2 = p0, p1, p2, p3
        else:
            l1, h1, l2, h2 = p0, p1, p2, p3

        atr_val = _atr_at(atr, p3.index)

        # Highs descending or flat
        if h2.price > h1.price + 0.15 * atr_val:
            continue
        # Lows ascending or flat
        if l2.price < l1.price - 0.15 * atr_val:
            continue

        # Range contraction (must be contracting significantly)
        w1 = h1.price - l1.price
        w2 = h2.price - l2.price
        if w2 >= w1 - 0.15 * atr_val:
            continue

        # Spacing check between consecutive pivots (ensure macro structure, not too close)
        leg_space = config.min_leg_spacing
        if (p1.index - p0.index < leg_space or
            p2.index - p1.index < leg_space or
            p3.index - p2.index < leg_space):
            continue

        neckline = (h2.price + l2.price) / 2.0

        if config.confirm_breakout:
            confirm_idx = None
            search_end = min(p3.index + config.breakout_window + 1, len(close))
            for bar in range(p3.index + 1, search_end):
                if close[bar] > h2.price or close[bar] < l2.price:
                    confirm_idx = bar
                    break
            if confirm_idx is None:
                continue
        else:
            confirm_idx = p3.index

        path = _forward_path(close, confirm_idx, config.horizon)
        if path is None:
            continue

        matches.append(PatternMatch(
            pattern="converging_triangle",
            pivots=[p0.index, p1.index, p2.index, p3.index],
            pivot_prices=[p0.price, p1.price, p2.price, p3.price],
            pivot_times=[p0.timestamp, p1.timestamp, p2.timestamp, p3.timestamp],
            neckline=neckline,
            confirm_index=confirm_idx,
            confirm_time=df["timestamp"].iloc[confirm_idx],
            forward_path=path,
            terminal_return=float(path[-1]),
        ))

    return matches


def _scan_diverging_triangle(
    df: pd.DataFrame,
    pivots: list[Pivot],
    config: PatternConfig,
    close: np.ndarray,
    atr: np.ndarray,
) -> list[PatternMatch]:
    """Diverging Triangle (发散三角形/扩张三角形): 6 alternating pivots with expanding range (3 highs and 3 lows).
    Alternates: H1, L1, H2, L2, H3, L3 or L1, H1, L2, H2, L3, H3
    Highs ascending/flat (H2 >= H1 - 0.15*ATR and H3 >= H2 - 0.15*ATR) and lows descending/flat (L2 <= L1 + 0.15*ATR and L3 <= L2 + 0.15*ATR),
    with overall expanding width: (H3 - L3) > (H2 - L2) + 0.02 * ATR and (H2 - L2) > (H1 - L1) + 0.02 * ATR, and (H3 - L3) > (H1 - L1) + 0.1 * ATR.
    """
    matches: list[PatternMatch] = []
    if len(pivots) < 6:
        return matches

    for i in range(len(pivots) - 5):
        p0, p1, p2, p3, p4, p5 = pivots[i : i + 6]
        dirs = [p0.direction, p1.direction, p2.direction, p3.direction, p4.direction, p5.direction]
        if dirs not in [
            ["high", "low", "high", "low", "high", "low"],
            ["low", "high", "low", "high", "low", "high"]
        ]:
            continue

        total_gap = p5.index - p0.index
        if not (config.min_gap * 1.5 <= total_gap <= config.max_gap * 2.5):
            continue

        if p0.direction == "high":
            h1, l1, h2, l2, h3, l3 = p0, p1, p2, p3, p4, p5
        else:
            l1, h1, l2, h2, l3, h3 = p0, p1, p2, p3, p4, p5

        atr_val = _atr_at(atr, p5.index)

        # Highs generally ascending/flat
        if h3.price < h1.price - 0.1 * atr_val:
            continue
        # Lows generally descending/flat
        if l3.price > l1.price + 0.1 * atr_val:
            continue

        # Range expansion (must be expanding significantly overall)
        w1 = h1.price - l1.price
        w2 = h2.price - l2.price
        w3 = h3.price - l3.price
        if w3 <= w1 + 0.12 * atr_val:
            continue
        if w3 <= w2 or w2 < w1 - 0.05 * atr_val:
            continue

        # Spacing check between consecutive pivots (ensure macro structure, not too close)
        leg_space = config.min_leg_spacing
        if (p1.index - p0.index < leg_space or
            p2.index - p1.index < leg_space or
            p3.index - p2.index < leg_space or
            p4.index - p3.index < leg_space or
            p5.index - p4.index < leg_space):
            continue

        neckline = (h3.price + l3.price) / 2.0

        if config.confirm_breakout:
            confirm_idx = None
            search_end = min(p5.index + config.breakout_window + 1, len(close))
            for bar in range(p5.index + 1, search_end):
                if close[bar] > h3.price or close[bar] < l3.price:
                    confirm_idx = bar
                    break
            if confirm_idx is None:
                continue
        else:
            confirm_idx = p5.index

        path = _forward_path(close, confirm_idx, config.horizon)
        if path is None:
            continue

        matches.append(PatternMatch(
            pattern="diverging_triangle",
            pivots=[p0.index, p1.index, p2.index, p3.index, p4.index, p5.index],
            pivot_prices=[p0.price, p1.price, p2.price, p3.price, p4.price, p5.price],
            pivot_times=[p0.timestamp, p1.timestamp, p2.timestamp, p3.timestamp, p4.timestamp, p5.timestamp],
            neckline=neckline,
            confirm_index=confirm_idx,
            confirm_time=df["timestamp"].iloc[confirm_idx],
            forward_path=path,
            terminal_return=float(path[-1]),
        ))

    return matches


def _scan_falling_wedge(
    df: pd.DataFrame,
    pivots: list[Pivot],
    config: PatternConfig,
    close: np.ndarray,
    atr: np.ndarray,
) -> list[PatternMatch]:
    """Falling Wedge: contracting highs and lows both sloping down.
    h1 > h2 (descending highs) and l1 > l2 (descending lows), with highs descending faster than lows.
    """
    matches: list[PatternMatch] = []
    if len(pivots) < 4:
        return matches

    for i in range(len(pivots) - 3):
        p0, p1, p2, p3 = pivots[i], pivots[i + 1], pivots[i + 2], pivots[i + 3]
        dirs = [p0.direction, p1.direction, p2.direction, p3.direction]
        if dirs not in [["high", "low", "high", "low"], ["low", "high", "low", "high"]]:
            continue

        total_gap = p3.index - p0.index
        if not (config.min_gap <= total_gap <= config.max_gap * 1.5):
            continue

        if p0.direction == "high":
            h1, l1, h2, l2 = p0, p1, p2, p3
        else:
            l1, h1, l2, h2 = p0, p1, p2, p3

        atr_val = _atr_at(atr, p3.index)

        # Descending highs and descending lows
        if not (h1.price > h2.price and (h1.price - h2.price) >= 0.12 * atr_val):
            continue
        if not (l1.price > l2.price and (l1.price - l2.price) >= 0.02 * atr_val):
            continue
        
        # Highs descend faster than lows (convergence)
        if not ((h1.price - h2.price) > (l1.price - l2.price)):
            continue
        if not ((h2.price - l2.price) < (h1.price - l1.price)):
            continue

        # Spacing check between consecutive pivots (ensure macro structure, not too close)
        leg_space = config.min_leg_spacing
        if (p1.index - p0.index < leg_space or
            p2.index - p1.index < leg_space or
            p3.index - p2.index < leg_space):
            continue

        neckline = (h2.price + l2.price) / 2.0

        if config.confirm_breakout:
            confirm_idx = None
            search_end = min(p3.index + config.breakout_window + 1, len(close))
            for bar in range(p3.index + 1, search_end):
                if close[bar] > h2.price or close[bar] < l2.price:
                    confirm_idx = bar
                    break
            if confirm_idx is None:
                continue
        else:
            confirm_idx = p3.index

        path = _forward_path(close, confirm_idx, config.horizon)
        if path is None:
            continue

        matches.append(PatternMatch(
            pattern="falling_wedge",
            pivots=[p0.index, p1.index, p2.index, p3.index],
            pivot_prices=[p0.price, p1.price, p2.price, p3.price],
            pivot_times=[p0.timestamp, p1.timestamp, p2.timestamp, p3.timestamp],
            neckline=neckline,
            confirm_index=confirm_idx,
            confirm_time=df["timestamp"].iloc[confirm_idx],
            forward_path=path,
            terminal_return=float(path[-1]),
        ))

    return matches


def _scan_rising_wedge(
    df: pd.DataFrame,
    pivots: list[Pivot],
    config: PatternConfig,
    close: np.ndarray,
    atr: np.ndarray,
) -> list[PatternMatch]:
    """Rising Wedge: contracting highs and lows both sloping up.
    l1 < l2 (ascending lows) and h1 < h2 (ascending highs), with lows ascending faster than highs.
    """
    matches: list[PatternMatch] = []
    if len(pivots) < 4:
        return matches

    for i in range(len(pivots) - 3):
        p0, p1, p2, p3 = pivots[i], pivots[i + 1], pivots[i + 2], pivots[i + 3]
        dirs = [p0.direction, p1.direction, p2.direction, p3.direction]
        if dirs not in [["high", "low", "high", "low"], ["low", "high", "low", "high"]]:
            continue

        total_gap = p3.index - p0.index
        if not (config.min_gap <= total_gap <= config.max_gap * 1.5):
            continue

        if p0.direction == "high":
            h1, l1, h2, l2 = p0, p1, p2, p3
        else:
            l1, h1, l2, h2 = p0, p1, p2, p3

        atr_val = _atr_at(atr, p3.index)

        # Ascending highs and ascending lows
        if not (l1.price < l2.price and (l2.price - l1.price) >= 0.12 * atr_val):
            continue
        if not (h1.price < h2.price and (h2.price - h1.price) >= 0.02 * atr_val):
            continue
        
        # Lows ascend faster than highs (convergence)
        if not ((l2.price - l1.price) > (h2.price - h1.price)):
            continue
        if not ((h2.price - l2.price) < (h1.price - l1.price)):
            continue

        # Spacing check between consecutive pivots (ensure macro structure, not too close)
        leg_space = config.min_leg_spacing
        if (p1.index - p0.index < leg_space or
            p2.index - p1.index < leg_space or
            p3.index - p2.index < leg_space):
            continue

        neckline = (h2.price + l2.price) / 2.0

        if config.confirm_breakout:
            confirm_idx = None
            search_end = min(p3.index + config.breakout_window + 1, len(close))
            for bar in range(p3.index + 1, search_end):
                if close[bar] > h2.price or close[bar] < l2.price:
                    confirm_idx = bar
                    break
            if confirm_idx is None:
                continue
        else:
            confirm_idx = p3.index

        path = _forward_path(close, confirm_idx, config.horizon)
        if path is None:
            continue

        matches.append(PatternMatch(
            pattern="rising_wedge",
            pivots=[p0.index, p1.index, p2.index, p3.index],
            pivot_prices=[p0.price, p1.price, p2.price, p3.price],
            pivot_times=[p0.timestamp, p1.timestamp, p2.timestamp, p3.timestamp],
            neckline=neckline,
            confirm_index=confirm_idx,
            confirm_time=df["timestamp"].iloc[confirm_idx],
            forward_path=path,
            terminal_return=float(path[-1]),
        ))

    return matches



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SCANNERS = {
    "w_bottom": _scan_w_bottom,
    "m_top": _scan_m_top,
    "head_shoulders": _scan_head_shoulders,
    "inv_head_shoulders": _scan_inv_head_shoulders,
    "converging_triangle": _scan_converging_triangle,
    "diverging_triangle": _scan_diverging_triangle,
    "falling_wedge": _scan_falling_wedge,
    "rising_wedge": _scan_rising_wedge,
}

PATTERN_LABELS = {
    "w_bottom": "Double Bottom (W)",
    "m_top": "Double Top (M)",
    "head_shoulders": "Head & Shoulders",
    "inv_head_shoulders": "Inverse Head & Shoulders",
    "converging_triangle": "Converging Triangle (收敛三角形)",
    "diverging_triangle": "Diverging Triangle (发散三角形)",
    "falling_wedge": "Falling Wedge (下降楔形)",
    "rising_wedge": "Rising Wedge (上升楔形)",
}


def scan_patterns(df: pd.DataFrame, config: PatternConfig) -> list[PatternMatch]:
    """Detect all instances of *config.pattern* in *df*.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`~a_shape_tool.core.load_ohlc_csv` with an
        additional ``atr`` column (added by
        :func:`~a_shape_tool.core.add_state_columns`).
    config:
        Pattern scan configuration.
    """
    if config.pattern not in _SCANNERS:
        raise ValueError(
            f"Unknown pattern '{config.pattern}'. "
            f"Choose from: {', '.join(_SCANNERS)}"
        )

    close = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    pivots = find_pivots(df, atr_mult=config.pivot_atr_mult)

    matches = _SCANNERS[config.pattern](df, pivots, config, close, atr)
    return matches


def make_pattern_quantiles(matches: list[PatternMatch]) -> pd.DataFrame:
    """Compute cross-sectional quantiles of all matched forward paths."""
    if not matches:
        return pd.DataFrame()
    paths = np.vstack([m.forward_path for m in matches])
    levels = [0.10, 0.25, 0.50, 0.75, 0.90]
    horizon = paths.shape[1] - 1
    rows = []
    for level in levels:
        values = np.quantile(paths, level, axis=0)
        row = {"quantile": level}
        row.update({f"t+{t}": float(values[t]) for t in range(horizon + 1)})
        rows.append(row)
    return pd.DataFrame(rows)


def matches_to_frame(matches: list[PatternMatch]) -> pd.DataFrame:
    """Convert matched instances to a summary DataFrame."""
    rows = []
    for rank, m in enumerate(matches, 1):
        rows.append({
            "rank": rank,
            "pattern": m.pattern,
            "confirm_time": m.confirm_time,
            "neckline": m.neckline,
            "terminal_return": m.terminal_return,
            "confirm_index": m.confirm_index,
            "n_pivots": len(m.pivots),
        })
    return pd.DataFrame(rows)

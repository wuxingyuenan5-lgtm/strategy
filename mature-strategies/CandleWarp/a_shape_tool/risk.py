"""risk.py — Probabilistic SL/TP targets and fractional Kelly Criterion sizing."""
from __future__ import annotations
from typing import Literal
import numpy as np

def calculate_probabilistic_risk(
    paths: np.ndarray,
    direction: Literal["long", "short"],
    fraction: float = 0.5,
) -> dict:
    """Calculate statistical Stop-Loss, Take-Profit, and recommended Kelly leverage
    based on historical forward return paths.

    Parameters
    ----------
    paths: np.ndarray of shape (M, horizon + 1)
        Historical matched forward return paths. Cumulative return since confirmation (path[0] = 0.0).
    direction: {"long", "short"}
        Whether this pattern signals an upside (long) or downside (short) breakout.
    fraction: float, default=0.5
        Fractional Kelly Criterion multiplier (e.g. 0.5 for half-Kelly).
    """
    if paths is None or len(paths) == 0:
        return {
            "win_rate": 0.0,
            "expected_value": 0.0,
            "tp_target": 0.0,
            "sl_target": 0.0,
            "reward_risk_ratio": 0.0,
            "kelly_leverage": 0.0,
            "edge": False
        }

    m, horizon = paths.shape
    
    # Calculate path high/low extremes during the horizon
    max_highs = np.max(paths, axis=1)        # Shape (M,)
    max_drawdowns = np.min(paths, axis=1)    # Shape (M,)

    if direction == "long":
        # TP: 75th percentile of max highs (bullish target)
        tp_target = float(np.quantile(max_highs, 0.75))
        # SL: 10th percentile of max drawdowns (bearish threshold)
        sl_target = float(np.quantile(max_drawdowns, 0.10))
        
        # Terminal win rate (finished positive)
        win_rate = float(np.mean(paths[:, -1] > 0.0))
        # Expected value
        expected_value = float(np.mean(paths[:, -1]))
    else: # short
        # TP: 25th percentile of max drawdowns (bearish target, represented as negative return)
        tp_target = float(np.quantile(max_drawdowns, 0.25))
        # SL: 90th percentile of max highs (bullish threshold, represented as positive return)
        sl_target = float(np.quantile(max_highs, 0.90))
        
        # Terminal win rate (finished negative)
        win_rate = float(np.mean(paths[:, -1] < 0.0))
        # Expected value (negative is winning for shorts, so negate it for the trader's payoff)
        expected_value = -float(np.mean(paths[:, -1]))

    # Ensure SL magnitude is non-zero to avoid division by zero
    sl_mag = abs(sl_target)
    tp_mag = abs(tp_target)
    
    if sl_mag < 1e-6:
        sl_mag = 0.001
        
    reward_risk_ratio = tp_mag / sl_mag
    
    # Compute Kelly Sizing
    # f* = p - (1-p)/b
    b = reward_risk_ratio
    p = win_rate
    
    if b > 0:
        f_star = p - (1.0 - p) / b
    else:
        f_star = 0.0
        
    # Scale by fractional Kelly multiplier
    kelly_leverage = fraction * f_star
    
    # Capital preservation constraints:
    # 1. No leverage if Kelly is negative (Expected Value < 0 or negative edge)
    # 2. Cap maximum leverage to 3.0x for conservative risk control
    if kelly_leverage < 0 or expected_value < 0:
        kelly_leverage = 0.0
        
    kelly_leverage = min(kelly_leverage, 3.0)
    
    return {
        "win_rate": round(win_rate * 100, 2),                  # as percentage e.g. 55.5
        "expected_value": round(expected_value * 100, 2),      # as percentage e.g. 0.42
        "tp_target": round(tp_target * 100, 2),                # as percentage e.g. +1.85
        "sl_target": round(sl_target * 100, 2),                # as percentage e.g. -0.65
        "reward_risk_ratio": round(reward_risk_ratio, 2),
        "kelly_leverage": round(kelly_leverage, 2),
        "edge": expected_value > 0 and f_star > 0
    }

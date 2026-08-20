from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .dtw import dtw_distance_2d
from .vp_fvg import compute_volume_profile, detect_unmitigated_fvg


OHLC_COLUMNS = ("open", "high", "low", "close")
TIME_ALIASES = ("timestamp", "time", "date", "datetime")

__all__ = [
    "SimilarityConfig",
    "SimilarityResult",
    "load_ohlc_csv",
    "run_similarity",
    "compute_atr",
    "compute_trend_bin",
    "compute_volatility_cuts",
    "apply_volatility_cuts",
    "compute_volatility_bin",
    "encode_window",
    "add_state_columns",
    "effective_min_match_gap",
    "resolve_query_end",
    "make_quantiles",
    "make_weighted_quantiles",
]


@dataclass(frozen=True)
class SimilarityConfig:
    timeframe: str = "1h"
    window: int = 100
    horizon: int = 50
    top_k: int = 50
    min_match_gap: int | None = None
    atr_period: int = 14
    trend_lookback: int = 120
    # >0 creates a meaningful flat state; 0.0 nearly eliminates it for hourly data
    flat_threshold: float = 0.01
    query_end: str = "last"
    history_only: bool = True
    # Scales body_ratio ∈ [-1,1] to be comparable with ATR-normalised features
    body_ratio_weight: float = 3.0
    # DTW extensions
    use_dtw: bool = True
    dtw_warping_window: int = 10
    dtw_rerank_k: int = 200
    # Risk Mitigation 1: Sample Size & Distance Gating
    min_valid_samples: int = 15
    max_distance_cutoff: float | None = None
    use_distance_weighting: bool = True
    # Microstructure extensions: Volume Profile & FVG
    use_volume: bool = False
    volume_weight: float = 1.0
    use_fvg: bool = False
    fvg_weight: float = 1.0


@dataclass(frozen=True)
class SimilarityResult:
    query: dict
    matches: pd.DataFrame
    paths: pd.DataFrame
    quantiles: pd.DataFrame
    weighted_quantiles: pd.DataFrame | None = None


def load_ohlc_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={column: column.strip().lower() for column in df.columns})

    missing = [column for column in OHLC_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required OHLC columns: {', '.join(missing)}")

    time_column = next((column for column in TIME_ALIASES if column in df.columns), None)
    if time_column:
        df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
        df = df.rename(columns={time_column: "timestamp"})
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        # Filter out weekends (Saturday is 5, Sunday is 6)
        is_weekend = df["timestamp"].dt.dayofweek.isin([5, 6])
        df = df[~is_weekend]
    else:
        df["timestamp"] = pd.RangeIndex(start=0, stop=len(df), step=1)

    for column in OHLC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        out_cols = ["timestamp", *OHLC_COLUMNS, "volume"]
    elif "vol" in df.columns:
        df["volume"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0.0)
        out_cols = ["timestamp", *OHLC_COLUMNS, "volume"]
    else:
        out_cols = ["timestamp", *OHLC_COLUMNS]

    df = df.dropna(subset=[*OHLC_COLUMNS]).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("CSV has too few valid OHLC rows.")

    return df[out_cols]


def run_similarity(df: pd.DataFrame, config: SimilarityConfig) -> SimilarityResult:
    _validate_config(config)
    base = df.copy().reset_index(drop=True)
    query_end = resolve_query_end(base, config.query_end)
    if config.history_only:
        base = base.iloc[: query_end + 1].copy().reset_index(drop=True)
        query_end = resolve_query_end(base, config.query_end)
    working = add_state_columns(base, config, query_end_index=query_end)

    query_start = query_end - config.window + 1

    if query_start < 0:
        raise ValueError("Query window starts before the first row; reduce --window.")

    query_state = working.loc[query_end, "state"]
    if pd.isna(query_state) or "unknown" in str(query_state):
        raise ValueError(
            "Query window has no complete state metadata. "
            "Use more data or reduce --atr-period / --trend-lookback."
        )

    query_vector = encode_window(
        working,
        query_start,
        query_end,
        body_ratio_weight=config.body_ratio_weight,
        use_volume=config.use_volume,
        volume_weight=config.volume_weight,
        use_fvg=config.use_fvg,
        fvg_weight=config.fvg_weight,
    )
    feature_dim = len(query_vector) // config.window

    candidates = iter_candidate_windows(working, query_start, query_end, config)

    rows: list[dict] = []
    path_rows: list[np.ndarray] = []
    for start, end in candidates:
        if working.loc[end, "state"] != query_state:
            continue

        try:
            vector = encode_window(
                working,
                start,
                end,
                body_ratio_weight=config.body_ratio_weight,
                use_volume=config.use_volume,
                volume_weight=config.volume_weight,
                use_fvg=config.use_fvg,
                fvg_weight=config.fvg_weight,
            )
        except ValueError:
            continue

        distance = float(np.linalg.norm(query_vector - vector))
        if config.max_distance_cutoff is not None and distance > config.max_distance_cutoff:
            continue

        future = forward_return_path(working, end, config.horizon)
        path_index = len(path_rows)
        rows.append(
            {
                "path_index": path_index,
                "start_index": start,
                "end_index": end,
                "start_time": working.loc[start, "timestamp"],
                "end_time": working.loc[end, "timestamp"],
                "distance": distance,
                "state": working.loc[end, "state"],
                "trend_bin": working.loc[end, "trend_bin"],
                "vol_bin": working.loc[end, "vol_bin"],
                "terminal_return": future[-1],
            }
        )
        path_rows.append(future)

    if not rows:
        raise ValueError(
            f"No historical candidates found in the same state ({query_state}). "
            "Try more history, a shorter window, or --no-history-only for research playback."
        )

    # Hierarchical Search: If use_dtw is enabled, pre-filter with Euclidean and re-rank with 2D-DTW
    if config.use_dtw:
        ranked_euclidean = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
        # Select the top dtw_rerank_k candidates for DTW evaluation
        dtw_candidates = ranked_euclidean.head(config.dtw_rerank_k).copy()

        query_matrix = query_vector.reshape(-1, feature_dim)
        dtw_rows = []
        for _, row in dtw_candidates.iterrows():
            start, end = int(row["start_index"]), int(row["end_index"])
            try:
                candidate_vector = encode_window(
                    working,
                    start,
                    end,
                    body_ratio_weight=config.body_ratio_weight,
                    use_volume=config.use_volume,
                    volume_weight=config.volume_weight,
                    use_fvg=config.use_fvg,
                    fvg_weight=config.fvg_weight,
                )
                candidate_matrix = candidate_vector.reshape(-1, feature_dim)
                dtw_dist = dtw_distance_2d(query_matrix, candidate_matrix, w=config.dtw_warping_window)
                row_dict = row.to_dict()
                row_dict["distance"] = dtw_dist
                dtw_rows.append(row_dict)
            except ValueError:
                continue

        ranked = pd.DataFrame(dtw_rows).sort_values("distance").reset_index(drop=True)
    else:
        ranked = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)

    selected = select_diverse_matches(ranked, config.top_k, effective_min_match_gap(config))
    if selected.empty:
        raise ValueError("No diverse matches found. Try reducing --min-match-gap.")
    matches = selected.drop(columns=["path_index"]).reset_index(drop=True)
    matches.insert(0, "rank", np.arange(1, len(matches) + 1))

    selected_paths = np.vstack([path_rows[int(i)] for i in selected["path_index"]])
    selected_distances = matches["distance"].to_numpy(dtype=float)

    path_columns = [f"t+{step}" for step in range(config.horizon + 1)]
    paths = pd.DataFrame(selected_paths, columns=path_columns)
    paths.insert(0, "rank", np.arange(1, len(paths) + 1))

    quantiles = make_quantiles(selected_paths)
    weighted_quantiles = (
        make_weighted_quantiles(selected_paths, selected_distances)
        if config.use_distance_weighting
        else None
    )

    found_count = len(matches)
    if found_count >= config.top_k:
        confidence = "HIGH"
    elif found_count >= config.min_valid_samples:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    query = {
        "query_start_index": query_start,
        "query_end_index": query_end,
        "query_start_time": working.loc[query_start, "timestamp"],
        "query_end_time": working.loc[query_end, "timestamp"],
        "state": query_state,
        "trend_bin": working.loc[query_end, "trend_bin"],
        "vol_bin": working.loc[query_end, "vol_bin"],
        "window": config.window,
        "horizon": config.horizon,
        "timeframe": config.timeframe,
        "atr_period": config.atr_period,
        "top_k_requested": config.top_k,
        "top_k_found": found_count,
        "confidence_level": confidence,
        "min_valid_samples": config.min_valid_samples,
        "min_match_gap": effective_min_match_gap(config),
        "mean_match_distance": float(np.mean(selected_distances)),
        "min_match_distance": float(np.min(selected_distances)),
    }
    return SimilarityResult(
        query=query,
        matches=matches,
        paths=paths,
        quantiles=quantiles,
        weighted_quantiles=weighted_quantiles,
    )


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def add_state_columns(
    df: pd.DataFrame,
    config: SimilarityConfig,
    query_end_index: int | None = None,
) -> pd.DataFrame:
    working = df.copy().reset_index(drop=True)
    working["atr"] = compute_atr(working, config.atr_period)
    working["trend_bin"] = compute_trend_bin(
        working["close"],
        lookback=config.trend_lookback,
        flat_threshold=config.flat_threshold,
    )
    atr_pct = (working["atr"] / working["close"]).replace([np.inf, -np.inf], np.nan)
    if query_end_index is not None and config.history_only:
        # Causal volatility quantile threshold estimation
        cuts = compute_volatility_cuts(atr_pct.iloc[: query_end_index + 1])
    else:
        cuts = compute_volatility_cuts(atr_pct)
    working["vol_bin"] = apply_volatility_cuts(atr_pct, cuts)
    working["state"] = working["trend_bin"] + "/" + working["vol_bin"]
    return working



def compute_trend_bin(close: pd.Series, lookback: int, flat_threshold: float) -> pd.Series:
    if lookback <= 0:
        return pd.Series("all", index=close.index)

    trend_return = close / close.shift(lookback) - 1.0
    values = np.select(
        [
            trend_return > flat_threshold,
            trend_return < -flat_threshold,
        ],
        ["up", "down"],
        default="flat",
    )
    result = pd.Series(values, index=close.index)
    result[trend_return.isna()] = "unknown"
    return result


def compute_volatility_cuts(atr_pct: pd.Series) -> tuple[float, float] | None:
    """Return (low_cut, high_cut) 33/66-percentile thresholds, or None if too few values."""
    valid = atr_pct.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3:
        return None
    low_cut, high_cut = valid.quantile([0.33, 0.66])
    return float(low_cut), float(high_cut)


def apply_volatility_cuts(
    atr_pct: pd.Series, cuts: tuple[float, float] | None
) -> pd.Series:
    """Classify each bar into lowvol/midvol/highvol using pre-computed thresholds."""
    if cuts is None:
        return pd.Series("unknown", index=atr_pct.index)
    low_cut, high_cut = cuts
    result = pd.Series("midvol", index=atr_pct.index)
    result[atr_pct <= low_cut] = "lowvol"
    result[atr_pct >= high_cut] = "highvol"
    result[atr_pct.isna()] = "unknown"
    return result


def compute_volatility_bin(atr_pct: pd.Series) -> pd.Series:
    """Convenience wrapper: compute cuts from the series itself (single-query mode)."""
    return apply_volatility_cuts(atr_pct, compute_volatility_cuts(atr_pct))


def encode_window(
    df: pd.DataFrame,
    start: int,
    end: int,
    body_ratio_weight: float = 3.0,
    use_volume: bool = False,
    volume_weight: float = 1.0,
    use_fvg: bool = False,
    fvg_weight: float = 1.0,
) -> np.ndarray:
    """Encode one OHLC(V) window into a flat feature vector of shape (window * D,).

    Base Features per bar (5 dims):
    - rel_open, rel_high, rel_low, rel_close (ATR-normalised relative to window[0].close)
    - body_ratio * body_ratio_weight

    Optional Microstructure extensions:
    - rel_volume (normalized relative volume intensity)
    - fvg_bias, nearest_fvg_dist (Fair Value Gap active state)
    """
    window = df.iloc[start : end + 1]
    if window.empty:
        raise ValueError("Empty window cannot be encoded.")

    atr = float(df.loc[end, "atr"])
    if not np.isfinite(atr) or atr <= 0:
        raise ValueError("Window has invalid ATR denominator.")

    reference_close = float(window["close"].iloc[0])
    rel_open = (window["open"].to_numpy(dtype=float) - reference_close) / atr
    rel_high = (window["high"].to_numpy(dtype=float) - reference_close) / atr
    rel_low = (window["low"].to_numpy(dtype=float) - reference_close) / atr
    rel_close = (window["close"].to_numpy(dtype=float) - reference_close) / atr

    candle_range = (window["high"] - window["low"]).replace(0, np.nan)
    body_ratio = (
        ((window["close"] - window["open"]) / candle_range).fillna(0.0).to_numpy(dtype=float)
        * body_ratio_weight
    )

    feature_cols = [rel_open, rel_high, rel_low, rel_close, body_ratio]

    if use_volume:
        if "volume" in window.columns and (window["volume"] > 0).any():
            from .vp_fvg import compute_volume_profile
            vp_res = compute_volume_profile(window, atr)
            poc_seq = np.full(len(window), vp_res.poc_rel * volume_weight, dtype=float)
            vah_seq = np.full(len(window), vp_res.vah_rel * volume_weight, dtype=float)
            val_seq = np.full(len(window), vp_res.val_rel * volume_weight, dtype=float)
            skew_seq = np.full(len(window), vp_res.vp_skew * volume_weight, dtype=float)
            feature_cols.extend([poc_seq, vah_seq, val_seq, skew_seq])
        else:
            feature_cols.extend([np.zeros(len(window), dtype=float) for _ in range(4)])

    if use_fvg:
        from .vp_fvg import detect_unmitigated_fvg
        fvg_res = detect_unmitigated_fvg(window, atr)
        bias_seq = np.full(len(window), fvg_res.net_fvg_bias * fvg_weight, dtype=float)
        dist_seq = np.full(len(window), fvg_res.nearest_fvg_dist * fvg_weight, dtype=float)
        feature_cols.extend([bias_seq, dist_seq])

    encoded = np.column_stack(feature_cols)
    if not np.isfinite(encoded).all():
        raise ValueError("Window encoding contains non-finite values.")

    return encoded.ravel()



def iter_candidate_windows(
    df: pd.DataFrame,
    query_start: int,
    query_end: int,
    config: SimilarityConfig,
) -> Iterable[tuple[int, int]]:
    max_start = len(df) - config.window - config.horizon
    for start in range(max_start + 1):
        end = start + config.window - 1
        if config.history_only:
            if end >= query_start:
                continue
            if end + config.horizon > query_end:
                continue
        if not config.history_only and ranges_overlap(start, end, query_start, query_end):
            continue
        yield start, end


def effective_min_match_gap(config: SimilarityConfig) -> int:
    if config.min_match_gap is not None:
        return max(0, int(config.min_match_gap))
    return max(config.window, config.horizon)


def select_diverse_matches(ranked: pd.DataFrame, top_k: int, min_gap: int) -> pd.DataFrame:
    if min_gap <= 0:
        return ranked.head(top_k)

    selected_rows = []
    selected_ends: list[int] = []
    for _, row in ranked.iterrows():
        end_index = int(row["end_index"])
        if all(abs(end_index - selected_end) >= min_gap for selected_end in selected_ends):
            selected_rows.append(row)
            selected_ends.append(end_index)
            if len(selected_rows) >= top_k:
                break

    if not selected_rows:
        return ranked.head(0)
    return pd.DataFrame(selected_rows)


def forward_return_path(df: pd.DataFrame, window_end: int, horizon: int) -> np.ndarray:
    future = df["close"].iloc[window_end : window_end + horizon + 1].to_numpy(dtype=float)
    if len(future) != horizon + 1:
        raise ValueError("Not enough future bars for candidate window.")
    return future / future[0] - 1.0


def make_quantiles(paths: np.ndarray) -> pd.DataFrame:
    levels = [0.10, 0.25, 0.50, 0.75, 0.90]
    rows = []
    for level in levels:
        values = np.quantile(paths, level, axis=0)
        row = {"quantile": level}
        row.update({f"t+{step}": float(value) for step, value in enumerate(values)})
        rows.append(row)
    return pd.DataFrame(rows)


def make_weighted_quantiles(
    paths: np.ndarray,
    distances: np.ndarray,
    tau: float | None = None,
) -> pd.DataFrame:
    """Compute distance-weighted quantiles using Softmax exponential decay weights."""
    n_samples, n_steps = paths.shape
    if n_samples == 0:
        return pd.DataFrame()

    dists = np.asarray(distances, dtype=float)
    if tau is None or tau <= 0:
        med_d = np.median(dists)
        tau = float(med_d) if med_d > 1e-4 else 1.0

    # Softmax weights: higher weight for smaller distances
    norm_d = (dists - np.min(dists)) / tau
    exp_w = np.exp(-norm_d)
    weights = exp_w / np.sum(exp_w)

    levels = [0.10, 0.25, 0.50, 0.75, 0.90]
    rows = []
    for level in levels:
        quant_vals = np.empty(n_steps, dtype=float)
        for t in range(n_steps):
            vals = paths[:, t]
            sorter = np.argsort(vals)
            sorted_vals = vals[sorter]
            sorted_weights = weights[sorter]
            cum_w = np.cumsum(sorted_weights)
            # Find interpolated value at quantile level
            quant_vals[t] = float(np.interp(level, cum_w, sorted_vals))

        row = {"quantile": level}
        row.update({f"t+{step}": float(val) for step, val in enumerate(quant_vals)})
        rows.append(row)

    return pd.DataFrame(rows)


def resolve_query_end(df: pd.DataFrame, query_end: str) -> int:
    if query_end == "last":
        return len(df) - 1

    if query_end.isdigit() or (query_end.startswith("-") and query_end[1:].isdigit()):
        index = int(query_end)
        if index < 0:
            index = len(df) + index
        if index < 0 or index >= len(df):
            raise ValueError(f"--query-end index out of range: {query_end}")
        return index

    timestamp = pd.to_datetime(query_end, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("--query-end must be 'last', an integer row index, or a timestamp.")

    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        eligible = df.index[df["timestamp"] <= timestamp]
        if len(eligible) == 0:
            raise ValueError("--query-end timestamp is before the first row.")
        return int(eligible[-1])

    raise ValueError("Timestamp query requires a CSV timestamp/time/date/datetime column.")


def ranges_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start <= right_end and right_start <= left_end


def _validate_config(config: SimilarityConfig) -> None:
    if config.window < 2:
        raise ValueError("--window must be at least 2.")
    if config.horizon < 1:
        raise ValueError("--horizon must be at least 1.")
    if config.top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if config.atr_period < 2:
        raise ValueError("--atr-period must be at least 2.")
    if config.min_valid_samples < 2:
        raise ValueError("--min-valid-samples must be at least 2.")


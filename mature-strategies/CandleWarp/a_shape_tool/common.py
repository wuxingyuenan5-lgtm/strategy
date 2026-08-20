"""common.py — Most-frequently-recurring pattern finder.

Instead of querying against the LATEST window, this module scans ALL
historical windows of a given size, finds the ones that cluster most
densely together (i.e. appear most often), and reports their forward-
return distribution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import (
    SimilarityConfig,
    add_state_columns,
    effective_min_match_gap,
    make_quantiles,
)
from .evaluation import make_feature_matrix, select_diverse_indices


@dataclass
class CommonPatternResult:
    """One cluster (archetype) of frequently-recurring windows."""
    cluster_id: int
    window: int
    centroid: np.ndarray          # feature vector of the cluster centre
    member_indices: np.ndarray    # end-bar indices of member windows (in df)
    member_distances: np.ndarray  # distance of each member from centroid
    member_times: list            # end-bar timestamps
    paths: pd.DataFrame           # (top_k × horizon+1) forward-return paths
    quantiles: pd.DataFrame       # 10/25/50/75/90 quantiles
    terminal_returns: np.ndarray
    frequency_count: int          # exact number of historical windows in this cluster
    frequency_pct: float          # percentage of all valid historical windows
    hourly_stats: pd.DataFrame    # 24 rows containing: hour, median, q25, q75, win_rate


def run_most_common(
    df: pd.DataFrame,
    config: SimilarityConfig,
    n_clusters: int = 3,
    state_filter: str | None = None,
) -> list[CommonPatternResult]:
    """Find the *n_clusters* most common recurring window shapes in *df* using K-Means.

    Parameters
    ----------
    df:
        OHLC DataFrame (output of :func:`~a_shape_tool.core.load_ohlc_csv`).
    config:
        Similarity config — ``window``, ``horizon``, ``top_k``,
        ``body_ratio_weight``, ``atr_period`` are used.
    n_clusters:
        How many distinct archetypes to extract.
    state_filter:
        Optional state string (e.g. "down/highvol") to restrict clustering to.
    """
    from sklearn.cluster import KMeans

    working = add_state_columns(df, config)
    close = working["close"].to_numpy(dtype=float)

    features = make_feature_matrix(working, config.window, config.body_ratio_weight)
    n = len(working)

    # Valid end-bar indices: window fits AND horizon fits
    max_end = n - config.horizon - 1
    candidate_ends = np.arange(config.window - 1, max_end + 1)
    valid_mask = np.isfinite(features[candidate_ends]).all(axis=1)
    valid_ends = candidate_ends[valid_mask]
    
    if state_filter is not None:
        state_mask = (working.loc[valid_ends, "state"].to_numpy() == state_filter)
        valid_ends = valid_ends[state_mask]

    valid_features = features[valid_ends]
    total_valid = len(valid_ends)

    if total_valid < config.top_k:
        state_msg = f" in state '{state_filter}'" if state_filter else ""
        raise ValueError(
            f"Not enough valid windows{state_msg} (found {total_valid}, need {config.top_k}). "
            "Reduce --window, --horizon, or --top-k, or choose a different state."
        )

    # Run KMeans to partition the shapes into structurally distinct clusters
    actual_clusters = min(n_clusters, total_valid)
    
    # Performance Optimization: sample if dataset is too large to fit in reasonable time
    if len(valid_features) > 50000:
        np.random.seed(42)
        sample_idx = np.random.choice(len(valid_features), 50000, replace=False)
        kmeans = KMeans(n_clusters=actual_clusters, n_init=10, random_state=42)
        kmeans.fit(valid_features[sample_idx])
        cluster_labels = kmeans.predict(valid_features)
        centroids = kmeans.cluster_centers_
    else:
        kmeans = KMeans(n_clusters=actual_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(valid_features)
        centroids = kmeans.cluster_centers_

    min_gap = effective_min_match_gap(config)
    results: list[CommonPatternResult] = []

    for c in range(actual_clusters):
        member_mask = (cluster_labels == c)
        c_ends = valid_ends[member_mask]
        c_features = valid_features[member_mask]
        c_count = int(np.sum(member_mask))
        c_pct = float(c_count / total_valid) * 100.0

        if len(c_ends) == 0:
            continue

        centroid = centroids[c]
        distances = np.linalg.norm(c_features - centroid, axis=1)

        # Select diverse top-K members
        top_indices = select_diverse_indices(
            c_ends, distances, config.top_k, min_gap
        )
        if len(top_indices) == 0:
            continue

        # Recalculate distances for the selected diverse members
        top_distances = np.linalg.norm(
            features[top_indices] - centroid, axis=1
        )

        # Build forward-return paths for the full config.horizon
        path_list = []
        for end_idx in top_indices:
            segment = close[end_idx: end_idx + config.horizon + 1]
            if len(segment) == config.horizon + 1 and segment[0] > 0:
                path_list.append(segment / segment[0] - 1.0)

        if not path_list:
            continue

        path_arr = np.vstack(path_list)
        cols = [f"t+{t}" for t in range(config.horizon + 1)]
        paths_df = pd.DataFrame(path_arr * 100.0, columns=cols)
        paths_df.insert(0, "rank", np.arange(1, len(paths_df) + 1))

        quantiles = make_quantiles(path_arr)

        # ── Micro-Analysis: Calculate discrete hourly stats for T+1 to T+24 ──
        # Gather 24h paths specifically
        h24_paths_list = []
        for end_idx in top_indices:
            segment24 = close[end_idx: min(n, end_idx + 24 + 1)]
            if len(segment24) > 1 and segment24[0] > 0:
                # pad with nan if it hits the end of file early
                rets24 = segment24 / segment24[0] - 1.0
                if len(rets24) < 25:
                    padding = np.full(25 - len(rets24), np.nan)
                    rets24 = np.concatenate([rets24, padding])
                h24_paths_list.append(rets24[1:]) # exclude T+0 (which is 0.0)

        h24_stats_rows = []
        if h24_paths_list:
            h24_matrix = np.vstack(h24_paths_list) * 100.0 # convert to %
            for h in range(1, 25):
                col_data = h24_matrix[:, h-1]
                valid_col_data = col_data[np.isfinite(col_data)]
                if len(valid_col_data) > 0:
                    med = float(np.median(valid_col_data))
                    q25 = float(np.percentile(valid_col_data, 25))
                    q75 = float(np.percentile(valid_col_data, 75))
                    win_rate = float((valid_col_data > 0.0).mean()) * 100.0
                else:
                    med, q25, q75, win_rate = 0.0, 0.0, 0.0, 0.0
                h24_stats_rows.append({
                    "hour": h,
                    "median": med,
                    "q25": q25,
                    "q75": q75,
                    "win_rate": win_rate
                })
        h24_stats_df = pd.DataFrame(h24_stats_rows)

        results.append(CommonPatternResult(
            cluster_id=c + 1,
            window=config.window,
            centroid=centroid,
            member_indices=top_indices,
            member_distances=top_distances,
            member_times=[
                working["timestamp"].iloc[i] for i in top_indices
            ],
            paths=paths_df,
            quantiles=quantiles,
            terminal_returns=path_arr[:, -1],
            frequency_count=c_count,
            frequency_pct=c_pct,
            hourly_stats=h24_stats_df,
        ))

    # Sort results by occurrence count descending so the most frequent appears first!
    results = sorted(results, key=lambda x: x.frequency_count, reverse=True)
    
    # Re-assign cluster IDs 1, 2, 3... based on frequency ranking
    for idx, r in enumerate(results):
        r.cluster_id = idx + 1

    return results


def optimize_kmeans_clusters(
    df: pd.DataFrame,
    config: SimilarityConfig,
    state_filter: str | None = None,
) -> dict[int, float]:
    """Calculate clustering inertia (Elbow Curve) for K in [2, 8]."""
    from sklearn.cluster import KMeans

    working = add_state_columns(df, config)
    features = make_feature_matrix(working, config.window, config.body_ratio_weight)
    n = len(working)

    max_end = n - config.horizon - 1
    candidate_ends = np.arange(config.window - 1, max_end + 1)
    valid_mask = np.isfinite(features[candidate_ends]).all(axis=1)
    valid_ends = candidate_ends[valid_mask]
    
    if state_filter is not None:
        state_mask = (working.loc[valid_ends, "state"].to_numpy() == state_filter)
        valid_ends = valid_ends[state_mask]

    valid_features = features[valid_ends]
    total_valid = len(valid_ends)
    
    inertias = {}
    max_k = min(8, total_valid)
    
    # Performance Optimization: sample for elbow curve if dataset is very large
    if len(valid_features) > 20000:
        np.random.seed(42)
        sample_idx = np.random.choice(len(valid_features), 20000, replace=False)
        fit_features = valid_features[sample_idx]
    else:
        fit_features = valid_features
        
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(fit_features)
        inertias[k] = float(kmeans.inertia_)
        
    return inertias

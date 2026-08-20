from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from numpy.lib.stride_tricks import sliding_window_view

from .core import (
    SimilarityConfig,
    apply_volatility_cuts,
    compute_atr,
    compute_trend_bin,
    compute_volatility_cuts,
    effective_min_match_gap,
)
from .dtw import dtw_distance_2d


@dataclass(frozen=True)
class BacktestConfig:
    similarity: SimilarityConfig
    min_history: int = 1000
    stride: int | None = None
    cost_bps: float = 0.0
    edge_threshold_bps: float = 0.0
    max_trials: int | None = None
    n_jobs: int = 1


@dataclass(frozen=True)
class BacktestResult:
    trials: pd.DataFrame
    summary: dict
    buckets: pd.DataFrame
    skipped: int
    skip_reasons: dict


def _evaluate_single_trial(
    query_end: int,
    working: pd.DataFrame,
    features: np.ndarray,
    terminal_returns: np.ndarray,
    sim: SimilarityConfig,
    config: BacktestConfig,
    min_required: int,
    effective_min_gap: int,
    n_channels: int,
) -> tuple[dict | None, str | None]:
    """Evaluate a single walk-forward query window with zero lookahead."""
    query_vector = features[query_end]
    query_state = working.loc[query_end, "state"]
    if not np.isfinite(query_vector).all() or "unknown" in str(query_state):
        return None, "invalid_state"

    query_start = query_end - sim.window + 1
    candidate_end_max = min(query_start - 1, query_end - sim.horizon)
    if candidate_end_max < sim.window - 1:
        return None, "not_enough_history"

    candidate_indices = np.arange(sim.window - 1, candidate_end_max + 1)
    valid = (
        np.isfinite(features[candidate_indices]).all(axis=1)
        & np.isfinite(terminal_returns[candidate_indices])
        & (working.loc[candidate_indices, "state"].to_numpy() == query_state)
    )
    candidate_indices = candidate_indices[valid]
    if len(candidate_indices) < min_required:
        return None, "not_enough_candidates"

    # Fast Euclidean Pre-filter
    distances = np.linalg.norm(features[candidate_indices] - query_vector, axis=1)

    # Apply DTW re-ranking if enabled (dynamic n_channels, not hardcoded)
    if sim.use_dtw:
        pre_top_count = min(len(candidate_indices), sim.dtw_rerank_k)
        pre_order = np.argsort(distances)[:pre_top_count]
        pre_candidates = candidate_indices[pre_order]
        query_matrix = query_vector.reshape(sim.window, n_channels)

        dtw_dists = np.empty(pre_top_count, dtype=float)
        for idx, cand_end in enumerate(pre_candidates):
            cand_matrix = features[cand_end].reshape(sim.window, n_channels)
            dtw_dists[idx] = dtw_distance_2d(query_matrix, cand_matrix, w=sim.dtw_warping_window)

        top_indices = select_diverse_indices(
            pre_candidates,
            dtw_dists,
            sim.top_k,
            effective_min_gap,
        )
        top_distances = dtw_dists[np.isin(pre_candidates, top_indices)]
    else:
        top_indices = select_diverse_indices(
            candidate_indices,
            distances,
            sim.top_k,
            effective_min_gap,
        )
        top_distances = distances[np.isin(candidate_indices, top_indices)]

    if len(top_indices) < min_required:
        return None, "not_enough_diverse"

    top_returns = terminal_returns[top_indices]
    state_returns = terminal_returns[candidate_indices]
    sim_terminal = quantile_dict(top_returns)
    baseline = quantile_dict(state_returns)
    baseline["count"] = int(len(state_returns))
    actual_return = float(terminal_returns[query_end])

    # Asymmetry Ratio: (Q75 - Q50) / (Q50 - Q25) with rigorous boundary handling
    upside_span = sim_terminal["q75"] - sim_terminal["q50"]
    downside_span = sim_terminal["q50"] - sim_terminal["q25"]

    if upside_span <= 1e-8 and downside_span <= 1e-8:
        asymmetry_ratio = 1.0
    elif downside_span <= 1e-8:
        asymmetry_ratio = 10.0 if upside_span > 0 else 1.0  # Dominant upside skew
    elif upside_span <= 1e-8:
        asymmetry_ratio = 0.1  # Dominant downside skew
    else:
        asymmetry_ratio = float(np.clip(upside_span / downside_span, 0.01, 100.0))

    edge_thr = config.edge_threshold_bps / 10000.0
    cost_frac = config.cost_bps / 10000.0

    # Rule 1: Standard Conservative Quantile Rule
    side_std = trade_side(sim_terminal["q25"], sim_terminal["q75"], threshold=edge_thr)
    net_return_std = side_std * actual_return - (cost_frac if side_std != 0 else 0.0)

    # Rule 2: Asymmetric Risk-Reward Filter Rule
    side_asym = trade_side_asymmetric(
        sim_terminal["q25"],
        sim_terminal["q50"],
        sim_terminal["q75"],
        threshold=edge_thr,
        cost_threshold=cost_frac,
        asymmetry_ratio=asymmetry_ratio,
    )
    net_return_asym = side_asym * actual_return - (cost_frac if side_asym != 0 else 0.0)

    confidence = "HIGH" if len(top_indices) >= sim.top_k else ("MEDIUM" if len(top_indices) >= sim.min_valid_samples else "LOW")

    row = {
        "query_end_index": query_end,
        "query_end_time": working.loc[query_end, "timestamp"],
        "state": query_state,
        "matches": int(len(top_indices)),
        "confidence": confidence,
        "actual_return": actual_return,
        "sim_q10": sim_terminal["q10"],
        "sim_q25": sim_terminal["q25"],
        "sim_median": sim_terminal["q50"],
        "sim_q75": sim_terminal["q75"],
        "sim_q90": sim_terminal["q90"],
        "asymmetry_ratio": asymmetry_ratio,
        "state_q25": baseline["q25"],
        "state_median": baseline["q50"],
        "state_q75": baseline["q75"],
        "state_candidates": baseline["count"],
        "sim_median_abs_error": abs(actual_return - sim_terminal["q50"]),
        "state_median_abs_error": abs(actual_return - baseline["q50"]),
        "covered_25_75": sim_terminal["q25"] <= actual_return <= sim_terminal["q75"],
        "covered_10_90": sim_terminal["q10"] <= actual_return <= sim_terminal["q90"],
        "direction_hit": same_sign(sim_terminal["q50"], actual_return),
        "state_direction_hit": same_sign(baseline["q50"], actual_return),
        "trade_side": side_std,
        "strategy_return": net_return_std,
        "trade_side_asym": side_asym,
        "strategy_return_asym": net_return_asym,
    }
    return row, None


def run_walk_forward(df: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    sim = config.similarity
    stride = config.stride or sim.horizon
    start_index = max(
        config.min_history,
        sim.window + sim.horizon + sim.atr_period + max(sim.trend_lookback, 0),
    )
    stop_index = len(df) - sim.horizon
    if stop_index <= start_index:
        raise ValueError("Not enough rows for walk-forward evaluation.")

    working = prepare_walk_forward_frame(df, sim, start_index)
    features = make_feature_matrix(
        working,
        sim.window,
        sim.body_ratio_weight,
        use_volume=sim.use_volume,
        use_fvg=sim.use_fvg,
    )
    terminal_returns = make_terminal_returns(working["close"].to_numpy(dtype=float), sim.horizon)

    rows: list[dict] = []
    skipped = 0
    skip_reasons: dict[str, int] = {
        "invalid_state": 0,
        "not_enough_history": 0,
        "not_enough_candidates": 0,
        "not_enough_diverse": 0,
    }

    min_required = sim.min_valid_samples if sim.min_valid_samples > 0 else sim.top_k
    gap = effective_min_match_gap(sim)
    n_channels = features.shape[1] // sim.window

    query_ends = list(range(start_index, stop_index, stride))
    if config.max_trials is not None:
        query_ends = query_ends[: config.max_trials]

    # Parallel or Sequential Execution
    import os
    from concurrent.futures import ThreadPoolExecutor

    actual_workers = (
        os.cpu_count() or 4 if config.n_jobs == -1 else max(1, config.n_jobs)
    )

    if actual_workers > 1 and len(query_ends) > 8:
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_single_trial,
                    q_end,
                    working,
                    features,
                    terminal_returns,
                    sim,
                    config,
                    min_required,
                    gap,
                    n_channels,
                )
                for q_end in query_ends
            ]
            for f in futures:
                row, reason = f.result()
                if row is not None:
                    rows.append(row)
                else:
                    skipped += 1
                    if reason in skip_reasons:
                        skip_reasons[reason] += 1
    else:
        for query_end in query_ends:
            row, reason = _evaluate_single_trial(
                query_end,
                working,
                features,
                terminal_returns,
                sim,
                config,
                min_required,
                gap,
                n_channels,
            )
            if row is not None:
                rows.append(row)
            else:
                skipped += 1
                if reason in skip_reasons:
                    skip_reasons[reason] += 1

    trials = pd.DataFrame(rows)
    if trials.empty:
        raise ValueError("No valid walk-forward trials. Try reducing --min-history or --top-k.")

    buckets = make_bucket_table(trials)
    summary = summarize_trials(trials, config, skipped)
    summary["skip_reasons"] = skip_reasons
    return BacktestResult(
        trials=trials,
        summary=summary,
        buckets=buckets,
        skipped=skipped,
        skip_reasons=skip_reasons,
    )


def run_multi_dataset_walk_forward(
    datasets: dict[str, pd.DataFrame],
    config: BacktestConfig,
    n_workers: int | None = None,
) -> dict[str, BacktestResult]:
    """Run parallel walk-forward evaluations concurrently across multiple assets/timeframes."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    results: dict[str, BacktestResult] = {}
    actual_workers = min(
        len(datasets), os.cpu_count() or 4 if n_workers in (None, -1) else n_workers
    )

    if actual_workers > 1 and len(datasets) > 1:
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            future_to_name = {
                executor.submit(run_walk_forward, df, config): name
                for name, df in datasets.items()
            }
            for future in future_to_name:
                name = future_to_name[future]
                results[name] = future.result()
    else:
        for name, df in datasets.items():
            results[name] = run_walk_forward(df, config)

    return results


def summarize_multi_dataset_results(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Produce a cross-asset comparison table of walk-forward evaluation metrics."""
    rows = []
    for name, res in results.items():
        s = res.summary
        rows.append({
            "Asset / Dataset": name,
            "Trials": s["trials"],
            "MAE Impv (%)": f"{s['median_mae_improvement_pct'] * 100.0:+.2f}%",
            "Spearman IC": f"{s['spearman_ic']:.4f}",
            "IC_IR": f"{s.get('ic_ir', 0.0):.3f}",
            "IC t-stat": f"{s.get('ic_t_stat', 0.0):.2f}",
            "25-75% Calib": f"{s['coverage_25_75'] * 100.0:.1f}%",
            "Dir Hit Rate": f"{s['direction_hit_rate'] * 100.0:.1f}%",
            "Trade Win Rate": f"{s['trade_win_rate'] * 100.0:.1f}%" if s["trades"] > 0 else "N/A",
            "Profit Factor": f"{s['profit_factor']:.2f}" if s["trades"] > 0 else "N/A",
            "Asym Trades": s.get("asym_trades", 0),
            "Asym Win Rate": f"{s.get('asym_trade_win_rate', 0.0) * 100.0:.1f}%" if s.get("asym_trades", 0) > 0 else "N/A",
        })
    return pd.DataFrame(rows)




def prepare_walk_forward_frame(
    df: pd.DataFrame,
    config: SimilarityConfig,
    first_query_index: int,
) -> pd.DataFrame:
    working = df.copy().reset_index(drop=True)
    working["atr"] = compute_atr(working, config.atr_period)
    working["trend_bin"] = compute_trend_bin(
        working["close"],
        lookback=config.trend_lookback,
        flat_threshold=config.flat_threshold,
    )

    atr_pct = (working["atr"] / working["close"]).replace([np.inf, -np.inf], np.nan)
    cuts = compute_volatility_cuts(atr_pct.iloc[: first_query_index + 1])
    working["vol_bin"] = apply_volatility_cuts(atr_pct, cuts)

    working["state"] = working["trend_bin"] + "/" + working["vol_bin"]
    return working


def make_feature_matrix(
    df: pd.DataFrame,
    window: int,
    body_ratio_weight: float = 3.0,
    use_volume: bool = False,
    use_fvg: bool = False,
) -> np.ndarray:
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    n = len(df)

    n_channels = 5
    has_vol = use_volume and "volume" in df.columns
    if has_vol:
        n_channels += 4
    if use_fvg:
        n_channels += 2

    features = np.full((n, window * n_channels), np.nan, dtype=float)
    if n < window:
        return features

    open_w = sliding_window_view(open_, window)
    high_w = sliding_window_view(high, window)
    low_w = sliding_window_view(low, window)
    close_w = sliding_window_view(close, window)

    ends = np.arange(window - 1, n)
    denom = atr[ends]
    ref = close_w[:, 0]

    candle_range = high_w - low_w
    body_ratio = (
        np.where(candle_range != 0, (close_w - open_w) / candle_range, 0.0)
        * body_ratio_weight
    )
    rel_open = (open_w - ref[:, None]) / denom[:, None]
    rel_high = (high_w - ref[:, None]) / denom[:, None]
    rel_low = (low_w - ref[:, None]) / denom[:, None]
    rel_close = (close_w - ref[:, None]) / denom[:, None]

    channels = [rel_open, rel_high, rel_low, rel_close, body_ratio]

    if has_vol:
        from .vp_fvg import compute_volume_profile
        vp_poc = np.empty((len(ends), window), dtype=float)
        vp_vah = np.empty((len(ends), window), dtype=float)
        vp_val = np.empty((len(ends), window), dtype=float)
        vp_skew = np.empty((len(ends), window), dtype=float)
        for i, end_i in enumerate(ends):
            vp = compute_volume_profile(
                df.iloc[end_i - window + 1 : end_i + 1],
                atr[end_i],
            )
            vp_poc[i, :] = vp.poc_rel
            vp_vah[i, :] = vp.vah_rel
            vp_val[i, :] = vp.val_rel
            vp_skew[i, :] = vp.vp_skew
        channels.extend([vp_poc, vp_vah, vp_val, vp_skew])

    if use_fvg:
        from .vp_fvg import detect_unmitigated_fvg
        fvg_bias = np.empty((len(ends), window), dtype=float)
        fvg_dist = np.empty((len(ends), window), dtype=float)
        for i, end_i in enumerate(ends):
            fvg = detect_unmitigated_fvg(
                df.iloc[end_i - window + 1 : end_i + 1],
                atr[end_i],
            )
            fvg_bias[i, :] = fvg.net_fvg_bias
            fvg_dist[i, :] = fvg.nearest_fvg_dist
        channels.extend([fvg_bias, fvg_dist])


    encoded = np.stack(channels, axis=2).reshape(len(ends), -1)
    valid = np.isfinite(denom) & (denom > 0) & np.isfinite(encoded).all(axis=1)
    features[ends[valid]] = encoded[valid]
    return features



def make_terminal_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    returns = np.full(len(close), np.nan, dtype=float)
    returns[: len(close) - horizon] = close[horizon:] / close[: len(close) - horizon] - 1.0
    return returns


def quantile_dict(values: np.ndarray) -> dict:
    q10, q25, q50, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "q10": float(q10),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "q90": float(q90),
    }


def select_diverse_indices(
    candidate_indices: np.ndarray,
    distances: np.ndarray,
    top_k: int,
    min_gap: int,
) -> np.ndarray:
    order = np.argsort(distances)
    sorted_candidates = candidate_indices[order]
    if min_gap <= 0:
        return sorted_candidates[:top_k]

    buffer = np.empty(top_k, dtype=int)
    n_sel = 0
    for candidate in sorted_candidates:
        if n_sel == 0 or np.all(np.abs(candidate - buffer[:n_sel]) >= min_gap):
            buffer[n_sel] = candidate
            n_sel += 1
            if n_sel >= top_k:
                break
    return buffer[:n_sel]


def terminal_quantiles(quantiles: pd.DataFrame, horizon: int) -> dict:
    column = f"t+{horizon}"
    values = quantiles.set_index("quantile")[column]
    return {
        "q10": float(values.loc[0.10]),
        "q25": float(values.loc[0.25]),
        "q50": float(values.loc[0.50]),
        "q75": float(values.loc[0.75]),
        "q90": float(values.loc[0.90]),
    }


def trade_side(q25: float, q75: float, threshold: float) -> int:
    if q25 > threshold:
        return 1
    if q75 < -threshold:
        return -1
    return 0


def trade_side_asymmetric(
    q25: float,
    q50: float,
    q75: float,
    threshold: float,
    cost_threshold: float,
    asymmetry_ratio: float,
    min_asym_long: float = 1.25,
    max_asym_short: float = 0.80,
) -> int:
    """Asymmetric Risk-Reward trade filter:
    - Long: Q25 covers execution cost, median positive, and upside elasticity exceeds downside (ratio >= 1.25).
    - Short: Q75 covers execution cost, median negative, and downside risk exceeds upside (ratio <= 0.80).
    """
    if q50 > threshold and q25 > -cost_threshold and asymmetry_ratio >= min_asym_long:
        return 1
    if q50 < -threshold and q75 < cost_threshold and asymmetry_ratio <= max_asym_short:
        return -1
    return 0


def same_sign(left: float, right: float) -> bool:
    if left == 0 or right == 0:
        return False
    return np.sign(left) == np.sign(right)


def make_bucket_table(trials: pd.DataFrame) -> pd.DataFrame:
    unique_values = trials["sim_median"].nunique()
    bucket_count = min(5, unique_values)
    if bucket_count < 2:
        return pd.DataFrame()

    bucketed = trials.copy()
    bucketed["bucket"] = pd.qcut(bucketed["sim_median"], q=bucket_count, duplicates="drop")
    grouped = (
        bucketed.groupby("bucket", observed=True)
        .agg(
            count=("actual_return", "size"),
            avg_sim_median=("sim_median", "mean"),
            avg_actual_return=("actual_return", "mean"),
            median_actual_return=("actual_return", "median"),
            hit_rate=("direction_hit", "mean"),
        )
        .reset_index()
    )
    grouped["bucket"] = grouped["bucket"].astype(str)
    return grouped


def summarize_trials(trials: pd.DataFrame, config: BacktestConfig, skipped: int) -> dict:
    sim_mae = float(trials["sim_median_abs_error"].mean())
    state_mae = float(trials["state_median_abs_error"].mean())
    improvement = state_mae - sim_mae
    improvement_pct = improvement / state_mae if state_mae > 0 else np.nan

    # Spearman IC & IC_IR Statistics
    preds = trials["sim_median"].to_numpy(dtype=float)
    actuals = trials["actual_return"].to_numpy(dtype=float)
    spearman_res = stats.spearmanr(preds, actuals)
    spearman_ic = float(spearman_res.statistic) if pd.notna(spearman_res.statistic) else np.nan
    spearman_p = float(spearman_res.pvalue) if pd.notna(spearman_res.pvalue) else np.nan

    # Rolling IC for stability & IC_IR
    n_trials = len(trials)
    if n_trials >= 10:
        rolling_win = min(15, n_trials // 2)
        rolling_ics = []
        for i in range(rolling_win, n_trials + 1):
            sub_pred = preds[i - rolling_win : i]
            sub_act = actuals[i - rolling_win : i]
            if len(np.unique(sub_pred)) > 1 and len(np.unique(sub_act)) > 1:
                r_ic = stats.spearmanr(sub_pred, sub_act).statistic
                if pd.notna(r_ic):
                    rolling_ics.append(r_ic)
        if rolling_ics:
            ic_mean = float(np.mean(rolling_ics))
            ic_std = float(np.std(rolling_ics, ddof=1)) if len(rolling_ics) > 1 else 0.0
            ic_ir = float(ic_mean / (ic_std + 1e-8))
            ic_t_stat = float(ic_ir * np.sqrt(len(rolling_ics)))
        else:
            ic_mean, ic_std, ic_ir, ic_t_stat = spearman_ic, 0.0, np.nan, np.nan
    else:
        ic_mean, ic_std, ic_ir, ic_t_stat = spearman_ic, 0.0, np.nan, np.nan

    # Trade summaries for standard rule and asymmetric rule
    trade_rows_std = trials[trials["trade_side"] != 0].copy()
    trade_summary_std = summarize_trades(trade_rows_std, "strategy_return")
    trade_summary_std["trade_rate"] = float(len(trade_rows_std) / len(trials))

    trade_rows_asym = trials[trials["trade_side_asym"] != 0].copy()
    trade_summary_asym = summarize_trades(trade_rows_asym, "strategy_return_asym")
    trade_summary_asym["trade_rate_asym"] = float(len(trade_rows_asym) / len(trials))

    # High/Medium confidence subset metrics
    high_conf = trials[trials["confidence"].isin(["HIGH", "MEDIUM"])]
    high_conf_ic = (
        float(stats.spearmanr(high_conf["sim_median"], high_conf["actual_return"]).statistic)
        if len(high_conf) >= 5 and len(np.unique(high_conf["sim_median"])) > 1
        else np.nan
    )

    return {
        "timeframe": config.similarity.timeframe,
        "window": config.similarity.window,
        "horizon": config.similarity.horizon,
        "top_k": config.similarity.top_k,
        "min_history": config.min_history,
        "stride": config.stride or config.similarity.horizon,
        "cost_bps": config.cost_bps,
        "edge_threshold_bps": config.edge_threshold_bps,
        "trials": int(len(trials)),
        "skipped": int(skipped),
        "coverage_25_75": float(trials["covered_25_75"].mean()),
        "coverage_10_90": float(trials["covered_10_90"].mean()),
        "direction_hit_rate": float(trials["direction_hit"].mean()),
        "state_direction_hit_rate": float(trials["state_direction_hit"].mean()),
        "median_mae_similarity": sim_mae,
        "median_mae_state_baseline": state_mae,
        "median_mae_improvement": float(improvement),
        "median_mae_improvement_pct": float(improvement_pct),
        # IC & IC_IR Robustness Metrics
        "spearman_ic": spearman_ic,
        "spearman_p_value": spearman_p,
        "ic_ir": ic_ir,
        "ic_t_stat": ic_t_stat,
        "valid_confidence_ic": high_conf_ic,
        "valid_confidence_ratio": float(len(high_conf) / len(trials)),

        # Trading Performance
        **trade_summary_std,
        # Asymmetric Rule Performance
        "asym_trades": trade_summary_asym["trades"],
        "asym_win_rate": trade_summary_asym["trade_win_rate"],
        "asym_profit_factor": trade_summary_asym["profit_factor"],
        "asym_max_drawdown": trade_summary_asym["max_drawdown"],
        "asym_compounded_return": trade_summary_asym["compounded_return"],
    }


def summarize_trades(trades: pd.DataFrame, return_col: str = "strategy_return") -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "trade_rate": 0.0,
            "avg_trade_return": np.nan,
            "trade_win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
            "return_to_risk": np.nan,
            "compounded_return": 0.0,
        }

    returns = trades[return_col].astype(float)
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    std = returns.std(ddof=1)
    return_to_risk = returns.mean() / std if std and std > 0 else np.nan

    return {
        "trades": int(len(trades)),
        "trade_rate": np.nan,
        "avg_trade_return": float(returns.mean()),
        "trade_win_rate": float((returns > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else np.inf,
        "max_drawdown": float(drawdown.min()),
        "return_to_risk": float(return_to_risk) if pd.notna(return_to_risk) else np.nan,
        "compounded_return": float(equity.iloc[-1] - 1.0),
    }

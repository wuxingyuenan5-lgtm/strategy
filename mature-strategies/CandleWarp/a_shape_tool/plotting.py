from __future__ import annotations

import html
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure Matplotlib premium dark theme globally
plt.style.use("dark_background")
matplotlib.rcParams.update({
    "figure.facecolor": "#090d16",     # Deep obsidian blue-slate (matching Tailwind slate-950)
    "axes.facecolor": "#0d1321",       # Elegant dark navy-slate
    "axes.edgecolor": "#1e293b",       # Subtle border (slate-800)
    "axes.grid": True,
    "grid.color": "#1e293b",           # Very subtle grid lines
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "text.color": "#cbd5e1",           # Soft white-grey text (slate-300)
    "axes.labelcolor": "#94a3b8",      # Slate-400 for labels
    "xtick.color": "#64748b",          # Slate-500 for ticks
    "ytick.color": "#64748b",
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Roboto", "DejaVu Sans", "Arial"],
})

from .core import SimilarityResult, compute_atr
from .common import CommonPatternResult


# ── common-pattern plots ─────────────────────────────────────────────────────

def plot_common_patterns(
    df: pd.DataFrame,
    results: list[CommonPatternResult],
    output_path: str | Path,
    title_suffix: str = "",
) -> Path:
    """Plot shape overlays + forward-return distributions for common-pattern clusters.

    Each cluster gets one row:
    - Left panel : all member windows overlaid (ATR-normalised, centred at 0)
    - Right panel: 10/25/50/75/90 quantile band of the forward return
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(results)
    if n == 0:
        return output_path

    fig, axes = plt.subplots(n, 2, figsize=(14, 4.2 * n), dpi=150)
    if n == 1:
        axes = [axes]          # make always 2-D

    close = df["close"].to_numpy(dtype=float)

    # Pre-compute ATR for normalisation
    from .core import compute_atr as _compute_atr
    atr_series = _compute_atr(df, 14).to_numpy(dtype=float)

    colors_shape = "#475569"   # Darker slate-600 shape overlay for better contrast
    color_median = "#3b82f6"   # Vibrant neon blue
    color_band   = "#2563eb"   # Semi-translucent dark blue band
    color_outer  = "#64748b"   # Slate-500 for outer lines

    for row_idx, res in enumerate(results):
        ax_shape, ax_dist = axes[row_idx]
        window = res.window
        horizon = res.paths.shape[1] - 1   # paths has rank col removed below
        x_shape = np.arange(window)
        x_dist  = np.arange(horizon + 1)

        # ── left: shape overlay ──
        for end_idx in res.member_indices:
            start_idx = end_idx - window + 1
            seg_close = close[start_idx: end_idx + 1].astype(float)
            atr_val   = atr_series[end_idx]
            if not (np.isfinite(atr_val) and atr_val > 0):
                atr_val = np.nanstd(seg_close) or 1.0
            ref = seg_close[0]
            norm = (seg_close - ref) / atr_val
            ax_shape.plot(x_shape, norm, color=colors_shape, lw=0.8, alpha=0.25)

        # Draw centroid shape
        centroid_vec = res.centroid.reshape(window, -1)[:, 3]   # rel_close channel
        ax_shape.plot(x_shape, centroid_vec, color=color_median, lw=2.2,
                      label="Archetype (centroid)")
        ax_shape.axhline(0, color="#ffffff", lw=0.8, alpha=0.15)
        ax_shape.set_title(
            f"Cluster {res.cluster_id}  |  window={window}  |  "
            f"n={len(res.member_indices)} instances",
            fontsize=11,
            color="#f1f5f9"
        )
        ax_shape.set_xlabel("Bars in window")
        ax_shape.set_ylabel("ATR-normalised move")
        ax_shape.grid(True, color="#1e293b", lw=0.7)
        ax_shape.legend(frameon=False, fontsize=8)

        # ── right: forward-return distribution ──
        paths_np = res.paths.drop(columns=["rank"]).to_numpy(dtype=float)  # shape: (n, horizon+1), already ×100
        horizon = paths_np.shape[1] - 1
        x_dist  = np.arange(horizon + 1)
        q = res.quantiles.set_index("quantile")
        q10 = q.loc[0.10].to_numpy(dtype=float) * 100
        q25 = q.loc[0.25].to_numpy(dtype=float) * 100
        q50 = q.loc[0.50].to_numpy(dtype=float) * 100
        q75 = q.loc[0.75].to_numpy(dtype=float) * 100
        q90 = q.loc[0.90].to_numpy(dtype=float) * 100

        for path in paths_np:
            ax_dist.plot(x_dist, path, color=colors_shape, lw=0.7, alpha=0.20)
        ax_dist.fill_between(x_dist, q25, q75, color=color_band, alpha=0.25,
                             label="25%–75%")
        ax_dist.plot(x_dist, q50, color=color_median, lw=2.2, label="Median")
        ax_dist.plot(x_dist, q10, color=color_outer, lw=1.1, ls="--",
                     label="10% / 90%")
        ax_dist.plot(x_dist, q90, color=color_outer, lw=1.1, ls="--")
        ax_dist.axhline(0, color="#ffffff", lw=0.9, alpha=0.15)

        term_med   = q50[-1]
        term_q25   = q25[-1]
        term_q75   = q75[-1]
        win_pct    = float((res.terminal_returns > 0).mean() * 100)
        ax_dist.set_title(
            f"Forward return after cluster {res.cluster_id}  "
            f"(horizon={horizon})\n"
            f"Terminal median {term_med:+.2f}%  [{term_q25:+.2f}%, {term_q75:+.2f}%]  "
            f"up {win_pct:.0f}%",
            fontsize=10,
            color="#f1f5f9"
        )
        ax_dist.set_xlabel("Bars after window end")
        ax_dist.set_ylabel("Return (%)")
        ax_dist.grid(True, color="#1e293b", lw=0.7)
        ax_dist.legend(frameon=False, fontsize=8)

    suptitle = f"Most-common recurring patterns{' – ' + title_suffix if title_suffix else ''}"
    fig.suptitle(suptitle, fontsize=13, y=1.01, color="#f8fafc")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_distribution(result: SimilarityResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paths = result.paths.drop(columns=["rank"]).to_numpy(dtype=float) * 100.0
    quantiles = result.quantiles.set_index("quantile")
    x = np.arange(paths.shape[1])
    q10 = quantiles.loc[0.10].to_numpy(dtype=float) * 100.0
    q25 = quantiles.loc[0.25].to_numpy(dtype=float) * 100.0
    q50 = quantiles.loc[0.50].to_numpy(dtype=float) * 100.0
    q75 = quantiles.loc[0.75].to_numpy(dtype=float) * 100.0
    q90 = quantiles.loc[0.90].to_numpy(dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=160)
    for row in paths:
        ax.plot(x, row, color="#475569", linewidth=0.8, alpha=0.22)

    ax.fill_between(x, q25, q75, color="#2563eb", alpha=0.25, label="25%-75% band")
    ax.plot(x, q50, color="#3b82f6", linewidth=2.4, label="Median")
    ax.plot(x, q10, color="#64748b", linewidth=1.2, linestyle="--", label="10% / 90%")
    ax.plot(x, q90, color="#64748b", linewidth=1.2, linestyle="--")
    ax.axhline(0, color="#ffffff", linewidth=1.0, alpha=0.15)

    query = result.query
    terminal_median = q50[-1]
    terminal_q25 = q25[-1]
    terminal_q75 = q75[-1]
    title = (
        f"Forward return distribution after top {query['top_k_found']} similar windows "
        f"(timeframe={query['timeframe']}, window={query['window']}, horizon={query['horizon']})"
    )
    subtitle = (
        f"Query end: {query['query_end_time']} | State: {query['state']} | "
        f"Terminal median: {terminal_median:.2f}% "
        f"[{terminal_q25:.2f}%, {terminal_q75:.2f}%]"
    )
    ax.set_title(title, fontsize=13, pad=14, color="#f1f5f9")
    ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=9.5, color="#94a3b8")
    ax.set_xlabel(x_axis_label(query["timeframe"]))
    ax.set_ylabel("Close return from window end (%)")
    ax.grid(True, color="#1e293b", linewidth=0.8)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def x_axis_label(timeframe: str) -> str:
    normalized = timeframe.strip().lower()
    if "m" in normalized:
        return f"Bars ({timeframe}) after window end"
    if "h" in normalized:
        return f"Bars ({timeframe}) after window end"
    if "d" in normalized:
        return "Days after window end"
    return f"Bars ({timeframe}) after window end"


def plot_similarity_diagnostics(
    df: pd.DataFrame,
    result: SimilarityResult,
    output_path: str | Path,
    max_matches: int = 6,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = result.query
    working = df.copy().reset_index(drop=True)
    working["atr"] = compute_atr(working, int(query["atr_period"]))
    query_start = int(query["query_start_index"])
    query_end = int(query["query_end_index"])
    matches = result.matches.head(max_matches).copy()

    n_matches = len(matches)
    
    # Dynamically structure the layout gridspec based on matches count to prevent empty quadrants
    if n_matches == 0:
        fig_height = 5.5
        fig = plt.figure(figsize=(13.5, fig_height), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 0.8], hspace=0.4)
        match_cols = 1
    else:
        match_cols = min(3, n_matches)
        match_rows = (n_matches + match_cols - 1) // match_cols
        total_grid_rows = 2 + match_rows
        height_ratios = [1.25, 0.8] + [1.0] * match_rows
        fig_height = 5.2 + 3.0 * match_rows
        fig = plt.figure(figsize=(13.5, fig_height), dpi=150)
        gs = fig.add_gridspec(total_grid_rows, match_cols, height_ratios=height_ratios, hspace=0.45, wspace=0.24)

    # ── Top Panel: Overlay Check ──
    ax_overlay = fig.add_subplot(gs[0, :])
    query_norm = normalized_window(working, query_start, query_end)
    x = np.arange(len(query_norm["close"]))
    ax_overlay.plot(x, query_norm["close"], color="#ffffff", linewidth=2.8, label="Query")
    ax_overlay.fill_between(
        x,
        query_norm["low"],
        query_norm["high"],
        color="#ffffff",
        alpha=0.08,
        linewidth=0,
    )

    colors = ["#3b82f6", "#10b981", "#f59e0b", "#a855f7", "#f43f5e", "#06b6d4"] # Vibrant neon colors
    for i, row in enumerate(matches.itertuples(index=False)):
        match_norm = normalized_window(working, int(row.start_index), int(row.end_index))
        ax_overlay.plot(
            x,
            match_norm["close"],
            color=colors[i % len(colors)],
            linewidth=1.4,
            alpha=0.74,
            label=f"#{int(row.rank)}",
        )

    ax_overlay.axhline(0, color="#ffffff", linewidth=0.8, alpha=0.15)
    ax_overlay.set_title(
        f"Similarity check: query vs top {len(matches)} matches "
        f"(window={query['window']}, timeframe={query['timeframe']})",
        fontsize=13,
        color="#f1f5f9"
    )
    ax_overlay.set_ylabel("ATR-normalized move")
    ax_overlay.grid(True, color="#1e293b", linewidth=0.8)
    ax_overlay.legend(ncol=4, frameon=False, fontsize=8)

    # ── Second Panel: Distance bar chart ──
    ax_dist = fig.add_subplot(gs[1, :])
    top_dist = result.matches.head(20)
    ax_dist.bar(top_dist["rank"].astype(str), top_dist["distance"], color="#3b82f6", alpha=0.85)
    ax_dist.set_title("Top-20 distance ranking (DTW Euclidean distance)", fontsize=10, color="#f1f5f9")
    ax_dist.set_xlabel("Rank", color="#94a3b8")
    ax_dist.set_ylabel("Euclidean distance", color="#94a3b8")
    ax_dist.grid(True, axis="y", color="#1e293b", linewidth=0.8)

    # ── Lower Panels: Individual Side-by-Side Plots ──
    for idx, row in enumerate(matches.itertuples(index=False)):
        gs_row = 2 + idx // match_cols
        gs_col = idx % match_cols
        ax = fig.add_subplot(gs[gs_row, gs_col])
        
        match_norm = normalized_window(working, int(row.start_index), int(row.end_index))
        
        # Shade query and plot query
        ax.fill_between(x, query_norm["low"], query_norm["high"], color="#ffffff", alpha=0.06, linewidth=0)
        ax.plot(x, query_norm["close"], color="#ffffff", linewidth=1.8, alpha=0.45, label="Query")
        
        # Shade match and plot match
        ax.fill_between(
            x,
            match_norm["low"],
            match_norm["high"],
            color=colors[idx % len(colors)],
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(x, match_norm["close"], color=colors[idx % len(colors)], linewidth=1.8, label="Match")
        
        ax.axhline(0, color="#ffffff", linewidth=0.7, alpha=0.15)
        ax.set_title(
            f"#{int(row.rank)} dist={row.distance:.2f}\n{row.end_time}",
            fontsize=8.5,
            color="#cbd5e1"
        )
        ax.grid(True, color="#1e293b", linewidth=0.7)
        if idx == 0:
            ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def normalized_window(df: pd.DataFrame, start: int, end: int) -> dict[str, np.ndarray]:
    window = df.iloc[start : end + 1]
    denominator = float(df.loc[end, "atr"])
    if not np.isfinite(denominator) or denominator <= 0:
        denominator = float((window["high"].max() - window["low"].min()) or 1.0)
    reference_close = float(window["close"].iloc[0])
    return {
        "open": (window["open"].to_numpy(dtype=float) - reference_close) / denominator,
        "high": (window["high"].to_numpy(dtype=float) - reference_close) / denominator,
        "low": (window["low"].to_numpy(dtype=float) - reference_close) / denominator,
        "close": (window["close"].to_numpy(dtype=float) - reference_close) / denominator,
    }


def write_html_report(
    result: SimilarityResult,
    image_path: str | Path,
    output_path: str | Path,
    diagnostics_path: str | Path | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_rel = Path(image_path).name
    q = result.query
    terminal = result.quantiles[["quantile", f"t+{q['horizon']}"]].copy()
    terminal[f"t+{q['horizon']}"] = terminal[f"t+{q['horizon']}"] * 100.0
    terminal_column = f"t+{q['horizon']}"
    terminal_rows = "\n".join(
        "<tr>"
        f"<td>{row['quantile']:.0%}</td>"
        f"<td>{row[terminal_column]:.3f}%</td>"
        "</tr>"
        for _, row in terminal.iterrows()
    )
    match_rows = "\n".join(
        "<tr>"
        f"<td>{int(row.rank)}</td>"
        f"<td>{html.escape(str(row.start_time))}</td>"
        f"<td>{html.escape(str(row.end_time))}</td>"
        f"<td>{row.distance:.4f}</td>"
        f"<td>{row.terminal_return * 100.0:.3f}%</td>"
        "</tr>"
        for row in result.matches.head(20).itertuples(index=False)
    )

    diag_section = ""
    if diagnostics_path is not None:
        diag_rel = html.escape(Path(diagnostics_path).name)
        diag_section = (
            f'  <h2>Similarity Diagnostics</h2>\n'
            f'  <img src="{diag_rel}" alt="Similarity diagnostics">\n'
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A-Shape Similarity Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    h2 {{ font-size: 16px; margin-top: 28px; }}
    p {{ color: #475569; }}
    img {{ width: 100%; height: auto; border: 1px solid #e5e7eb; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #f8fafc; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin: 16px 0; }}
    .meta div {{ background: #f8fafc; padding: 10px 12px; border: 1px solid #e5e7eb; }}
  </style>
</head>
<body>
<main>
  <h1>A-Shape Similarity Report</h1>
  <p>Historical distribution view only. This report does not produce a trading signal.</p>
  <div class="meta">
    <div><strong>Timeframe</strong><br>{html.escape(str(q["timeframe"]))}</div>
    <div><strong>Query start</strong><br>{html.escape(str(q["query_start_time"]))}</div>
    <div><strong>Query end</strong><br>{html.escape(str(q["query_end_time"]))}</div>
    <div><strong>State</strong><br>{html.escape(str(q["state"]))}</div>
    <div><strong>Matches</strong><br>{q["top_k_found"]} / {q["top_k_requested"]}</div>
  </div>
  <h2>Forward Return Distribution</h2>
  <img src="{html.escape(image_rel)}" alt="Forward return distribution">
{diag_section}  <h2>Terminal Return Quantiles</h2>
  <table>
    <thead><tr><th>Quantile</th><th>Return at horizon</th></tr></thead>
    <tbody>{terminal_rows}</tbody>
  </table>
  <h2>Top Matches</h2>
  <table>
    <thead><tr><th>Rank</th><th>Start</th><th>End</th><th>Distance</th><th>Terminal return</th></tr></thead>
    <tbody>{match_rows}</tbody>
  </table>
</main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def draw_candlesticks_on_ax(ax: plt.Axes, segment: pd.DataFrame, width: float = 0.6) -> None:
    # Set premium candle colors
    color_up = "#10b981"   # Emerald green
    color_down = "#ef4444" # Rose red
    
    # We plot relative bar indices on the x-axis (0 to len(segment)-1)
    x = np.arange(len(segment))
    opens = segment["open"].to_numpy(dtype=float)
    highs = segment["high"].to_numpy(dtype=float)
    lows = segment["low"].to_numpy(dtype=float)
    closes = segment["close"].to_numpy(dtype=float)
    
    # Determine colors
    colors = np.where(closes >= opens, color_up, color_down)
    
    # Draw shadows
    ax.vlines(x, lows, highs, colors=colors, linewidth=1.0)
    
    # Draw bodies
    bottoms = np.minimum(opens, closes)
    heights = np.abs(opens - closes)
    
    # Prevent height=0 drawing issues by setting a tiny minimum height
    min_h = (highs - lows) * 0.02
    heights = np.where(heights == 0, np.where(min_h == 0, 0.01, min_h), heights)
    
    # Add Rectangle patches
    for xi, bottom, height, color in zip(x, bottoms, heights, colors):
        rect = plt.Rectangle((xi - width / 2.0, bottom), width, height, facecolor=color, edgecolor=color)
        ax.add_patch(rect)
        
    ax.set_xlim(-1, len(segment))
    # Adjust y-limits to show a small margin
    y_min, y_max = lows.min(), highs.max()
    margin = (y_max - y_min) * 0.05 or 1.0
    ax.set_ylim(y_min - margin, y_max + margin)


def plot_ohlc_candles_grid(
    df: pd.DataFrame,
    indices: list[int] | np.ndarray | pd.Series,
    window: int,
    horizon: int,
    output_path: str | Path,
    title: str,
    max_plots: int = 6,
    query_index: int | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_indices = len(indices)
    n_plots = min(n_indices + (1 if query_index is not None else 0), max_plots)
    if n_plots == 0:
        return output_path

    # Arrange them in a dynamic grid layout based on n_plots to avoid stretching and blank quadrants
    if n_plots == 1:
        cols, rows = 1, 1
        figsize = (8, 4.8)
    elif n_plots == 2:
        cols, rows = 2, 1
        figsize = (13, 5.0)
    elif n_plots == 3:
        cols, rows = 3, 1
        figsize = (15, 4.8)
    else:
        cols = 3
        rows = (n_plots + 2) // 3
        figsize = (15, 4.0 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=150)
    
    # Flatten axes for easy iteration
    if n_plots == 1:
        axes = np.array([axes])
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = np.array([axes]).ravel()

    plot_idx = 0

    if query_index is not None:
        ax = axes[plot_idx]
        start_idx = max(0, query_index - window + 1)
        segment = df.iloc[start_idx : query_index + 1]
        
        draw_candlesticks_on_ax(ax, segment)
        ax.set_title(
            f"QUERY | {segment['timestamp'].iloc[0]} to {segment['timestamp'].iloc[-1]}",
            fontsize=9.5, fontweight="bold", color="#f1f5f9"
        )
        ax.grid(True, color="#1e293b", lw=0.7)
        ax.set_ylabel("Price", color="#94a3b8")
        plot_idx += 1

    for i, end_idx in enumerate(indices):
        if plot_idx >= max_plots:
            break
        ax = axes[plot_idx]
        start_idx = max(0, end_idx - window + 1)
        # Slices up to end_idx + horizon
        segment = df.iloc[start_idx : min(len(df), end_idx + horizon + 1)]

        draw_candlesticks_on_ax(ax, segment)
        
        # Draw dotted line at end of window
        # The window ends at `window - 1` (0-indexed relative to start_idx)
        actual_window_len = end_idx - start_idx + 1
        ax.axvline(actual_window_len - 1, color="#3b82f6", linestyle="--", linewidth=1.2, alpha=0.8, label="End of Pattern")
        
        # Calculate return if horizon is available
        ref_close = float(df.loc[end_idx, "close"])
        term_idx = min(len(df) - 1, end_idx + horizon)
        term_close = float(df.loc[term_idx, "close"])
        term_return = (term_close / ref_close - 1.0) * 100.0
        
        ax.set_title(
            f"Match #{i+1} | {segment['timestamp'].iloc[0]} to {df.loc[end_idx, 'timestamp']}\n"
            f"Horizon Return: {term_return:+.2f}%", 
            fontsize=9.5, color="#cbd5e1"
        )
        ax.grid(True, color="#1e293b", lw=0.7)
        ax.set_ylabel("Price", color="#94a3b8")
        plot_idx += 1

    # Hide unused axes
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=12, y=0.99, fontweight="bold", color="#f8fafc")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_elbow_curve(inertias: dict[int, float], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ks = sorted(inertias.keys())
    values = [inertias[k] for k in ks]

    ax.plot(ks, values, marker="o", color="#3b82f6", linewidth=2.0, markersize=6, label="Inertia")
    ax.set_title("K-Means Cluster Optimization (Elbow Method)", fontsize=11, fontweight="bold", pad=12, color="#f1f5f9")
    ax.set_xlabel("Number of Clusters (K)", fontsize=9.5, color="#94a3b8")
    ax.set_ylabel("Sum of Squared Distances (Inertia)", fontsize=9.5, color="#94a3b8")
    ax.set_xticks(ks)
    ax.grid(True, color="#1e293b", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_hourly_micro_analysis(stats_df: pd.DataFrame, output_path: str | Path, title: str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if stats_df.empty:
        return output_path

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    ax2 = ax1.twinx()

    hours = stats_df["hour"].to_numpy(dtype=int)
    medians = stats_df["median"].to_numpy(dtype=float)
    q25 = stats_df["q25"].to_numpy(dtype=float)
    q75 = stats_df["q75"].to_numpy(dtype=float)
    win_rates = stats_df["win_rate"].to_numpy(dtype=float)

    # 1. Plot win rate as a subtle bar chart on the right axis
    # Cyan bar chart representing the hourly win rate %
    ax2.bar(hours, win_rates, color="#22d3ee", alpha=0.18, width=0.6, label="Win Rate (%)")
    ax2.set_ylabel("Hourly Win Rate (%)", color="#22d3ee", fontsize=9.5)
    ax2.tick_params(axis="y", labelcolor="#22d3ee")
    ax2.set_ylim(0, 100)
    # Add a horizontal line at 50% win rate
    ax2.axhline(50.0, color="#22d3ee", linestyle=":", linewidth=1.0, alpha=0.4)

    # 2. Plot Median & IQR on the left axis
    # Shaded band for IQR (25%-75%)
    ax1.fill_between(hours, q25, q75, color="#3b82f6", alpha=0.16, label="25%–75% Range (IQR)")
    # Solid blue line for the median return
    ax1.plot(hours, medians, color="#60a5fa", linewidth=2.2, label="Median Return (%)")
    ax1.axhline(0.0, color="#ffffff", linewidth=0.9, linestyle="-", alpha=0.15)

    ax1.set_title(title, fontsize=11, fontweight="bold", pad=12, color="#f1f5f9")
    ax1.set_xlabel("Hours after Matched Pattern", fontsize=9.5, color="#94a3b8")
    ax1.set_ylabel("Cumulative Return (%)", color="#60a5fa", fontsize=9.5)
    ax1.tick_params(axis="y", labelcolor="#60a5fa")
    ax1.set_xticks(hours)
    ax1.grid(True, color="#1e293b", linewidth=0.6)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path



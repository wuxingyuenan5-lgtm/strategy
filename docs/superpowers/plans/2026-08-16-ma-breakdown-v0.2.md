# MA Breakdown v0.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible A-share moving-average breakdown research pipeline that fetches index history, detects 3-day true breakdown/recovery cycles, computes v0.2 metrics and analytics, validates outputs, and renders the v0.1.3-style HTML report from generated data.

**Architecture:** Python pipeline with explicit configuration, raw/processed data layers, a deterministic event state machine, analysis modules, JSON/CSV payloads, and a Jinja2 renderer. GitHub Actions executes the full networked pipeline on pushes to the v0.2 branch and uploads the report bundle as an artifact.

**Tech Stack:** Python 3.12, pandas, numpy, requests, PyYAML, Jinja2, pytest, GitHub Actions.

## Global Constraints
- Baseline is v0.1.3; do not overwrite the archived baseline.
- Report data cutoff is 2026-07-31.
- Main event definition: three consecutive closes below MA = true breakdown; three consecutive closes above MA = true recovery.
- Main historical distributions include only cycles that complete R3.
- Prices, Peak, Trough, 甲/乙/丙 all use Close.
- Peak is highest Close in the 10 trading days before D1, excluding D1; duplicate peak uses latest occurrence.
- Trough is lowest Close from D1 through R3; duplicate trough uses earliest occurrence.
- Index set: 上证指数、上证50、沪深300、中证1000、创业板指、科创50、中证红利; remove 中证全指.
- Long-history sample floor is 2005-01-04; growth indices use their actual available start.
- Preserve v0.1.3 overall HTML visual framework and eight-page navigation; remove meaningless desktop nav scrollbar.
- Every number in HTML must come from generated payloads, never hand-entered research results.

---

### Task 1: Project scaffold and market-data contract
**Files:** create `ma_breakdown/config/instruments.yaml`, `ma_breakdown/config/research.yaml`, `ma_breakdown/src/data.py`, `ma_breakdown/tests/test_data.py`, `ma_breakdown/requirements.txt`.

**Interfaces:** `load_instruments(path) -> list[dict]`; `parse_eastmoney_klines(payload, instrument) -> pd.DataFrame`; `fetch_instrument(instrument, begin, end) -> pd.DataFrame`.

- [ ] Write failing tests for field mapping, date ordering, duplicate-date rejection, cutoff clipping, and empty API response.
- [ ] Run `pytest ma_breakdown/tests/test_data.py -v` and verify RED.
- [ ] Implement parser/fetcher and YAML configs.
- [ ] Run test and verify GREEN.
- [ ] Validate instrument identities and `secid` mappings in returned API metadata before accepting data.

### Task 2: Deterministic event state machine
**Files:** create `ma_breakdown/src/events.py`, `ma_breakdown/tests/test_events.py`.

**Interfaces:** `compute_ma(df, window) -> pd.Series`; `detect_candidate_stats(df, ma_col) -> dict`; `detect_events(df, ma_col, sample_start, cutoff) -> tuple[pd.DataFrame,pd.DataFrame]`.

- [ ] Write failing synthetic-path tests for D1/D2/D3, false breakdowns, 1-2 day false recoveries, R1/R2/R3, repeated cycles, ongoing right-censored events, Peak duplicate rule, Trough duplicate rule.
- [ ] Verify RED.
- [ ] Implement state machine.
- [ ] Verify GREEN and full tests.

### Task 3: Event metrics and duration analytics
**Files:** create `ma_breakdown/src/metrics.py`, `ma_breakdown/tests/test_metrics.py`.

**Interfaces:** `enrich_event_metrics(events, prices) -> pd.DataFrame`; `depth_summary(events) -> pd.DataFrame`; `duration_summary(events) -> dict`; `window_path_stats(events, prices, horizons=(3,5,10,20,40,60)) -> pd.DataFrame`.

- [ ] Write failing tests for 甲1/2/3, 乙, 丙(R1 only after R3 confirmation), P1 beyond R3, right-censored P1, time-to-trough/recovery, tail probabilities, fixed-window max drawdown.
- [ ] Verify RED; implement; verify GREEN.

### Task 4: Conditional risk, signal timing, and cross-index conduction
**Files:** create `ma_breakdown/src/analytics.py`, `ma_breakdown/tests/test_analytics.py`.

**Interfaces:** `add_risk_features(...)`; `select_conditional_panels(...)`; `signal_period_stats(...)`; `conduction_matrix(...)`.

- [ ] Write failing tests for D3 MA deviation, D1→D3, Peak→D1/D3, MA slope, MA60/120/200 state, realized vol, contemporaneous breadth, candidate success rates, and common-sample conduction denominators.
- [ ] Verify RED; implement; verify GREEN.
- [ ] Conditional panel selector keeps only variables with sufficient sample, ordered bucket coverage, and material monotonic/tail-risk separation.

### Task 5: Payload validation and HTML renderer
**Files:** create `ma_breakdown/src/render.py`, `ma_breakdown/src/validate.py`, `ma_breakdown/templates/report.html`, `ma_breakdown/tests/test_render.py`.

**Interfaces:** `build_payload(...) -> dict`; `validate_payload(payload) -> list[str]`; `render_report(payload, template_path, output_path)`.

- [ ] Write failing tests that the eight pages exist, research definitions are present, seven correct indices are listed, 中证全指 is absent, no desktop nav overflow rule remains, output contains no placeholder values, and event counts reconcile.
- [ ] Verify RED; implement v0.1.3-compatible styling and v0.2 page contents; verify GREEN.
- [ ] Embed compact SVG charts generated from payload data; no externally fetched chart data at view time.

### Task 6: End-to-end pipeline and GitHub Actions production
**Files:** create `ma_breakdown/run_pipeline.py`, `.github/workflows/ma-breakdown-v02.yml`, `ma_breakdown/tests/test_pipeline.py`.

**Interfaces:** `python ma_breakdown/run_pipeline.py --cutoff 2026-07-31 --output ma_breakdown/output/v0.2`.

- [ ] Write failing integration test using fixture data to require `index.html`, `events.csv`, `summary.json`, `data_quality.json`, and raw/processed manifests.
- [ ] Verify RED; implement pipeline; verify GREEN.
- [ ] Workflow installs dependencies, runs pytest, runs full networked pipeline, validates output, and uploads `ma-breakdown-v0.2-report` artifact.
- [ ] On real workflow run, inspect logs/data-quality checks; fail the workflow if an instrument has wrong identity, non-monotonic dates, duplicated dates, insufficient warm-up, cutoff mismatch, unreconciled event counts, or HTML placeholders.

### Task 7: Production verification and version handoff
**Files:** modify branch docs/CHANGELOG only after successful run; preserve main/archive.

- [ ] Download workflow artifact and independently inspect `summary.json`, `data_quality.json`, `events.csv`, and rendered HTML.
- [ ] Recompute selected events and rolling MA windows independently against raw CSVs.
- [ ] Confirm HTML visible content matches payload counts/statistics and opens correctly.
- [ ] Store v0.2 output bundle in the v0.2 branch, update changelog, and leave main untouched until user review.

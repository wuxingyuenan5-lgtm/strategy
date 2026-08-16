# MA Breakdown v0.2 Report Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the v0.2 report layer so that it preserves the mature v0.1.3 visual/research-report style while using only the newly rebuilt v0.2 data, event definitions, metrics, and analysis outputs.

**Architecture:** Keep the current Tencent data acquisition, event engine, metrics, and v0.2 event database as the only source of truth. Expand the analysis payload and narrative generator, then render through a v0.1.3-derived Jinja template. No v0.1.3 numeric result may be copied into v0.2; v0.1.3 is used only as a UI, information-architecture, and research-expression reference.

**Tech Stack:** Python 3.12, pandas, numpy, Jinja2, pytest, GitHub Actions, self-contained HTML/CSS/JS/SVG.

## Global Constraints

- Keep the 8 report pages: 研究概览、研究设计、跌幅深度分析、条件风险分析、持续时间分析、时段与信号图表、跨指数传导、事件明细。
- Preserve v0.1.3 visual language: 1100px content width, card hierarchy, typography, `table-scroll`, `hl`, `warn`, filters, signal chart presentation, and research-text density.
- Remove the top-navigation horizontal scrollbar on desktop; do not remove horizontal scrolling from genuinely wide data tables.
- All numeric content, examples, rankings, period conclusions, and narrative claims must be generated from v0.2 data only.
- Data cutoff remains 2026-07-31 for this version.
- Event universe and definitions remain the already-approved v0.2 rules: 3-day true breakdown, 3-day true recovery, Peak=D1 previous 10 trading-day max Close, Trough=D1..R3 min Close, 甲1/2/3, 乙, 丙, P1, completed-event main sample, fake breakdowns used only for success-rate analysis.
- Do not modify `main`; work only on `agent/ma-breakdown-v0.2`.

---

### Task 1: Add report-structure regression tests

**Files:**
- Modify: `ma_breakdown/tests/test_pipeline.py`
- Modify: `ma_breakdown/tests/test_render.py`

**Interfaces:**
- Consumes: `run_from_frames(...)` and `render_report(...)`.
- Produces: regression assertions protecting the v0.1.3-style report contract.

- [ ] Add tests asserting rendered HTML has all 8 page IDs, multiple research cards, `table-scroll`, `signal-wrap`, `filters`, `hl`, and `warn` components.
- [ ] Add a desktop navigation regression asserting no `overflow-x:auto` on `.nav` and no generic `overflow:auto` wrapper around normal tables.
- [ ] Add tests asserting each analytical page contains at least one explanatory paragraph/summary block after tables rather than pure data output.
- [ ] Add tests asserting old v0.1.3-specific entities/results such as `中证全指`, `1605`, and old 甲/乙 definitions do not appear.
- [ ] Run `PYTHONPATH=. pytest ma_breakdown/tests/test_render.py ma_breakdown/tests/test_pipeline.py -q` and confirm the new tests fail before implementation.

### Task 2: Expand v0.2 analytical payload for research-report presentation

**Files:**
- Modify: `ma_breakdown/src/metrics.py`
- Modify: `ma_breakdown/src/analytics.py`
- Modify: `ma_breakdown/run_pipeline.py`
- Test: `ma_breakdown/tests/test_metrics.py`
- Test: `ma_breakdown/tests/test_analytics.py`

**Interfaces:**
- Produces payload keys: `signal_reliability`, `depth_tables`, `depth_comparisons`, `extreme_events`, `duration_tables`, `duration_buckets`, `recovery_curves`, `period_summary`, `period_detail`, `conduction_summary`, `conduction_detail`, `signal_examples`.

- [ ] Add complete per-index/per-MA distributions for 甲1/甲2/甲3/乙/丙, including n, mean, median, std, P25/P75/P90/P95/P99, extreme, threshold probabilities, and positive-rate for 丙.
- [ ] Add Day1 and Day2 candidate-to-Day3 success-rate tables by index, MA, and market period; fake breakdowns remain excluded from the main event universe.
- [ ] Add duration distributions for D1→Trough, D3→Trough, Trough→R1, Trough→R3, D1→R3, and D1→P1, including censored P1 counts and fixed 3/5/10/20/40/60-day risk-path statistics.
- [ ] Add duration buckets `≤3`, `4–5`, `6–10`, `11–20`, `21–40`, `41–60`, `61–120`, `>120` where applicable.
- [ ] Add market-period tables using the v0.2 event database; period boundaries may reuse the v0.1.3 period labels as presentation taxonomy, but every statistic must be recomputed.
- [ ] Add extreme-event ranking based on v0.2 乙 and 甲1/甲3 metrics.
- [ ] Expand cross-index transmission results with direction, horizon, denominator, same-period observability, and strongest/weakest pairs.
- [ ] Select representative signal paths from actual v0.2 events for the signal-chart page.
- [ ] Run focused tests and then full `pytest`.

### Task 3: Build data-driven research narrative generation

**Files:**
- Create: `ma_breakdown/src/narrative.py`
- Create: `ma_breakdown/tests/test_narrative.py`
- Modify: `ma_breakdown/run_pipeline.py`

**Interfaces:**
- `build_narrative(payload_inputs: dict) -> dict`
- Produces: `overview_conclusions`, `depth_commentary`, `conditional_commentary`, `duration_commentary`, `period_commentary`, `conduction_commentary`, `method_notes`.

- [ ] Write tests requiring each narrative claim to contain numbers/entities present in the current payload and to avoid fixed historical v0.1.3 values.
- [ ] Generate 6–8 overview conclusions from current data: signal reliability, thick-tail risk, confirmation-cost change from 甲1→甲3, full-cycle 乙, MA20/25 differences, style/index differences, duration vs price recovery, and cross-index resonance.
- [ ] Generate per-section explanatory text after each major table, including caveats for small samples and censored P1 observations.
- [ ] Generate period-by-period research text from computed statistics rather than hardcoded conclusions.
- [ ] Generate conditional-risk interpretations only for selected panels that pass minimum sample and separation checks.
- [ ] Run `pytest ma_breakdown/tests/test_narrative.py -q` and full suite.

### Task 4: Restore v0.1.3 report design language in the Jinja template

**Files:**
- Modify: `ma_breakdown/templates/report.html`
- Modify: `ma_breakdown/src/render.py`
- Test: `ma_breakdown/tests/test_render.py`

**Interfaces:**
- Consumes the expanded v0.2 payload and narratives from Tasks 2–3.
- Produces a self-contained `index.html` preserving the v0.1.3 visual/report experience.

- [ ] Port the v0.1.3 CSS hierarchy and components: 1100px sections, cards, h2/h3/h4 hierarchy, `table-scroll`, filters, signal wrappers, legends, notes, highlights and warnings.
- [ ] Keep desktop navigation as one visible row without a scrollbar; on small screens allow compact wrapping.
- [ ] Rebuild 研究概览 with headline metrics plus 6–8 data-driven core conclusions.
- [ ] Rebuild 研究设计 with exact v0.2 definitions and validation tables; explicitly state all definitions approved in the discussion.
- [ ] Rebuild 跌幅深度分析 as MA20/MA25 full tables plus thick-tail discussion, index-style comparison, 甲1→甲3 confirmation-cost analysis, 乙 full drawdown, 丙 recovery result, MA20/25 comparison, and extreme events.
- [ ] Rebuild 条件风险分析 into multiple cards with the selected v0.2 condition panels and written interpretations.
- [ ] Rebuild 持续时间分析 around three clocks: risk formation, MA recovery, price recovery; include cumulative completion curves and 3/5/10/20/40/60-day risk paths.
- [ ] Rebuild 时段与信号图表 with period overview, period-by-period research commentary, filters, full period×index table, and actual signal-path charts.
- [ ] Rebuild 跨指数传导 with summary interpretation, matrices/rankings, horizon controls or tables, and denominator/caveat text.
- [ ] Rebuild 事件明细 with v0.1.3-style filters and the expanded v0.2 fields for Peak/Trough, 甲1/2/3/乙/丙, R1/R3, P1 and duration.
- [ ] Verify ordinary tables do not get meaningless scrollbars; only true wide tables use `table-scroll`.

### Task 5: Production rerun and independent verification

**Files:**
- Modify if needed: `.github/workflows/ma-breakdown-v02.yml`
- Generated: `ma_breakdown/output/v0.2/*`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces the final GitHub-archived v0.2 research bundle and downloadable artifact.

- [ ] Run the complete local/fixture test suite and require zero failures.
- [ ] Trigger GitHub Actions production run on `agent/ma-breakdown-v0.2`.
- [ ] Require: data quality PASS, all 7 current indices, cutoff 2026-07-31, no `中证全指`, nonzero completed event count, and all generated narrative sections present.
- [ ] Inspect workflow logs for exact test count and bundle verification.
- [ ] Download the production artifact and independently inspect `index.html`, `events.csv`, `summary.json`, `data_quality.json`, and `raw_manifest.json`.
- [ ] Spot-check several historical events and narrative claims against raw v0.2 data.
- [ ] Update CHANGELOG with the report-layer rework and explicitly state that v0.1.3 provided design reference only, not numeric inputs.
- [ ] Deliver the final HTML plus complete analysis ZIP to the user.

# Research Design Page v0.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite only the V0.2 “研究设计” page so it fully documents the chosen MA signal-cycle methodology, Peak/Trough/P1 definitions, hindsight nature of 丙, and auxiliary/conditional variables, without changing V0.2 data calculations or other report pages.

**Architecture:** Keep the existing analysis engine and report template structure. Add render-contract tests for the agreed methodology text, then replace only the `page-design` content in `ma_breakdown/templates/report.html`; run the full test suite and GitHub Actions historical production to regenerate the HTML from the same V0.2 data pipeline.

**Tech Stack:** Python 3.12, pytest, Jinja2, GitHub Actions.

## Global Constraints

- Keep branch `agent/ma-breakdown-v0.2`; do not merge `main`.
- Do not change event recognition, Peak/Trough formulas, metrics, samples, raw data, or other analysis pages.
- Formal statistical unit remains the MA signal cycle: 3 consecutive closes below MA confirms true breakdown; 3 consecutive closes at/above MA confirms true recovery.
- Discuss complete market drawdown cycles only as an alternative methodology and explicitly explain why they are not used for formal statistics.
- Peak remains the highest Close in the 10 trading days before D1, excluding D1.
- Trough remains the lowest Close from D1 through R3, inclusive.
- 丙 remains `R1 Close / D1 Close - 1` and must be labeled as hindsight/path analysis.
- P1 remains first future Close >= D1 Close, searched beyond R3 if necessary.
- Do not move the 1,505-event dependence/correlation discussion into research design;共振 belongs to conditional-risk analysis.

---

### Task 1: Lock the research-design content contract

**Files:**
- Modify: `ma_breakdown/tests/test_render.py`

**Interfaces:**
- Consumes: `render_report(dummy_payload(), template_path, out)`.
- Produces: assertions that the rendered design page contains all agreed methodology definitions.

- [ ] Add assertions for: `完整市场回撤周期`, `均线信号周期`, `不参与本报告的正式事件识别`, `Day1前10个交易日`, `[D1, R3]`, `后视镜`, `价格修复`, `趋势修复`, `P1`, `辅助变量`, `条件变量`, `是否包含未来信息`.
- [ ] Run `PYTHONPATH=. pytest ma_breakdown/tests/test_render.py -q` and confirm failure before template change.

### Task 2: Rewrite only the research-design page

**Files:**
- Modify: `ma_breakdown/templates/report.html`

**Interfaces:**
- Consumes: existing `research.ma_validation` and `instruments` payload fields.
- Produces: unchanged 8-page report with expanded methodology documentation.

- [ ] Replace the current `page-design` body with seven cards/sections: research-object choice; signal-cycle state machine; Peak/Trough; 甲乙丙; P1; auxiliary/conditional variables; calculation validation/data range.
- [ ] Explicitly state all price anchors use Close and tie rules (Peak last occurrence; Trough first occurrence).
- [ ] Use a horizontally scrollable variable-definition table and distinguish D3-known variables from hindsight/path variables.
- [ ] Keep calculation validation and data range tables intact.
- [ ] Run `PYTHONPATH=. pytest ma_breakdown/tests/test_render.py -q` and confirm pass.

### Task 3: Full regression and production regeneration

**Files:**
- Generated: `ma_breakdown/output/v0.2/index.html`

**Interfaces:**
- Consumes: unchanged V0.2 pipeline/data source.
- Produces: final regenerated HTML artifact.

- [ ] Run full `PYTHONPATH=. pytest ma_breakdown/tests -q` via GitHub Actions.
- [ ] Run full historical analysis through the existing workflow.
- [ ] Verify data quality remains PASS and event count remains 1,505 unless the unchanged upstream source itself changes.
- [ ] Download the successful workflow artifact and deliver the regenerated `index.html` to the user.

# MA Breakdown v0.2 Feature Parity Restoration Plan

> **For agentic workers:** execute task-by-task with verification gates.

**Goal:** Restore the analytical depth and interactive functions of v0.1.3 while keeping v0.2 data, event definitions, index pool, and conclusions as the sole research source.

**Architecture:** Keep the existing v0.2 data/event engine. Expand the research payload and renderer, not the raw event definitions. v0.1.3 is used only as a UI/information-architecture/functionality reference; no v0.1.3 statistics are copied.

**Tech Stack:** Python 3.12, pandas, numpy, Jinja2, inline SVG/JavaScript, GitHub Actions.

## Global Constraints

- Branch only: `agent/ma-breakdown-v0.2`; do not modify `main`.
- Data cutoff remains `2026-07-31`.
- Index pool remains 上证指数、上证50、沪深300、中证1000、创业板指、科创50、中证红利.
- Event engine remains 3-day true breakdown / 3-day true recovery with v0.2 Peak/Trough/甲1/甲2/甲3/乙/丙/P1 definitions.
- Old v0.1.3 numbers, event counts, findings, and old event definitions must never enter v0.2 payload or narrative.
- Wide analytical tables should preserve all useful columns and use explicit horizontal scrolling; do not drop metrics merely to fit the viewport.

---

### Task 1: Lock feature-parity tests

- Add renderer assertions for wide-table scroll containers, full depth tables, market-stage summary, interactive signal curve controls, conduction matrices, duration correlation/long-event sections, and richer event columns.
- Verify tests fail before implementation.

### Task 2: Restore full depth-analysis payload

- Expand per-index/MA statistics to include mean/median/std/P75-risk/P90/P95/P99/extreme/tail probabilities for 甲1、甲2、甲3、乙.
- Add 丙 full distribution and positive-rate metrics.
- Add multi-axis duration columns to the MA20/MA25 full-distribution tables.
- Restore style comparison, confirmation-cost, MA20-vs-MA25, and extreme-event analysis using v0.2 events.

### Task 3: Restore structured conditional-risk framework

- Add global tail-probability table.
- Add explicit D3 deviation buckets, D1→D3 decline buckets, breadth buckets, long-cycle trend-state buckets.
- Add MA20→MA25 escalation analysis.
- Add survival/state table showing how risk changes when an event remains unrecovered after 5/10/20/40 trading days.
- Keep automatically selected exploratory factors as a supplementary section rather than the whole page.

### Task 4: Restore full duration analysis

- Add full D1→R3 distribution by index/MA with mean/median/P75/P90/P95/max and completion-rate columns.
- Add duration-bucket vs 甲1/甲3/乙/丙 table.
- Add per-index/MA correlations between duration and risk metrics.
- Add longest-event table.
- Retain v0.2 risk-release / R3 / P1 curves and fixed 3/5/10/20/40/60-day risk path.

### Task 5: Restore market-stage analysis and interactive signal curve

- Stage overview should separate MA20 and MA25 rather than only aggregate both.
- Stage×index table should restore full statistical columns and actual coverage.
- Each stage gets data-driven narrative covering overall risk, style dispersion, MA differences, tail risk, and repair speed.
- Embed compact raw close/MA20/MA25 history for the 7 indices.
- Restore controls: stage / index / MA.
- Restore signal curve: close, selected MA, below-MA shading, true-breakdown Day1 marker, successful true-recovery R1 marker, R3 confirmation marker. Ongoing true breakdowns may be shown without recovery markers.

### Task 6: Restore conduction matrices and lag statistics

- Extend D3-confirmation based transmission calculations to 3/5/10/20 trading days.
- Record same-day rate and hit lag mean/median.
- Render MA20 and MA25 10-day matrices with median lag; expose shorter-window rates in cell metadata.
- Restore full directional table and data-driven transmission narrative.

### Task 7: Expand event detail and filters

- Restore explicit index/MA/date filters and reset button.
- Add Peak/D1/D2/D3/Trough/R1/R3/P1 dates and prices, D1/D2/D3 MA values, 甲1/2/3/乙/丙, all key durations, breadth and selected risk-state columns.
- Keep table horizontally scrollable rather than dropping columns.

### Task 8: Research design and validation

- Restore MA calculation validation table using v0.2 raw data.
- Restore data-range/source table and explicit caveats.
- Make the v0.2 definitions prominent and auditable.

### Task 9: End-to-end production verification

- Run full tests.
- Run GitHub Actions full historical analysis.
- Verify data quality PASS, all seven source histories reach 2026-07-31, event count matches event CSV, no old index/name/count text leaks into HTML.
- Verify interactive signal curve data is embedded and all major analytical tables have scroll containers.
- Download and inspect final artifact before delivery.

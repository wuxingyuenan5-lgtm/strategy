# 均线有效跌破统计 v0.1.3 — Baseline Snapshot

- Version: `v0.1.3`
- Baseline frozen: `2026-08-16`
- Report data cutoff: `2026-07-31`
- Original report title: `均线有效跌破统计 v0.1.3`
- Original artifact filename: `index.html`
- Original artifact last modified: `2026-08-07T08:22:38Z`
- Research universe: 7 A-share indices
- Moving averages: MA20 / MA25
- Confirmation rule: 3 consecutive closes below corresponding MA
- Total effective-break events: 1,605

## Frozen research state

This snapshot is the reference state immediately before the next round of complex research modifications.

Current modules:

1. 研究概览
2. 研究设计
3. 跌幅深度分析
4. 条件风险分析
5. 持续时间分析
6. 时段与信号图表
7. 跨指数传导
8. 事件明细

Key v0.1.3 additions relative to earlier versions:

- Day3 deviation-from-MA conditional risk buckets;
- multi-index breadth / synchronized-breakdown analysis;
- corrected CSI All Share identity to `000985.CSI`;
- explicit prohibition on stitching discontinuous CSI All Share history into MA event paths;
- improved wide-table scrolling and mobile presentation;
- strengthened distinction between MA-length effect and market-regime effect.

## Artifact preservation note

The exact original `index.html` artifact is retained in the ChatGPT File Library and is treated as the visual/report source of truth for v0.1.3. The GitHub baseline files freeze the research definitions, version state and conclusions so future iterations can be compared against this snapshot without overwriting it.

Do not reinterpret this baseline silently. Any material change to signal definition, formulas, universe, data identity, regime partition, conditional variables or conclusions must be recorded in `CHANGELOG.md` and released under a new version number.

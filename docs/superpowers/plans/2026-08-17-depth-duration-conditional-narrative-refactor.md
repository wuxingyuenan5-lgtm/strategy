# 跌幅深度 / 持续时间 / 条件风险叙事重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变V0.2任何事件定义、样本和原始数据的前提下，把“跌幅深度分析、持续时间分析、条件风险分析”改造成研究结论优先、统计底表后置的可读研究页，并把持续时间分析移动到条件风险分析之前。

**Architecture:** 直接复用当前冻结的 `events.csv` 与现有最终HTML，不重新抓取原始数据，不重新识别事件。新增一个本地审阅版patch脚本，仅重建三个子页的HTML内容与导航/页面顺序；派生统计全部由冻结事件表即时计算。最终审阅HTML单独留档到GitHub review路径，避免触发完整历史生产工作流。

**Tech Stack:** Python 3.12, pandas, inline SVG, HTML/CSS/JS.

## Global Constraints
- D1/D2/D3、R1/R2/R3、Peak、Trough、P1、甲1/甲2/甲3/乙/丙定义全部不变。
- 完整周期仍为1505个；进行中事件仍为12个。
- 不重新抓取腾讯数据，不重新运行事件识别流水线。
- `.table-scroll` 横向滚动功能保留；正式报告不出现“请横向滚动”“为适配屏幕”等操作性提示。
- 无方向比例（概率、标准差、成功率）不显示正号；方向收益/回撤保留正负号。
- 风险分位统一表述为“风险P90/P95/P99”。
- 页面顺序调整为：研究概览 → 研究设计 → 跌幅深度分析 → 持续时间分析 → 条件风险分析 → 时段与信号图表 → 跨指数传导 → 事件明细。

---

### Task 1: 重构跌幅深度分析

**Files:**
- Create local patch script: `/mnt/data/ma_v02_narrative_refactor.py`
- Consume: `/mnt/data/均线有效跌破统计_v0.2_跌幅深度优化版.html`
- Consume: `/mnt/data/ma_v02_final/events.csv`

**Interfaces:**
- Produces `build_depth_page(events) -> str`.

- [ ] 保留现有分位风险图、风格图、确认成本、风险路径拆解、丙、MA20/25、极端事件与完整宽表。
- [ ] 调整顺序为“问题与核心结论 → 风险深度 → 厚尾 → 三日确认 → 风险路径 → 风格 → 丙 → MA20/25 → 极端事件 → 完整底表”。
- [ ] 每个模块加入“数据现象 → 含义 → 边界/注意事项”的文字分析。
- [ ] 把MA20/MA25完整宽表移动到本页最后。

### Task 2: 重构持续时间分析

**Interfaces:**
- Produces `build_duration_page(events) -> str`.

- [ ] 先解释三条时间轴：D1→Trough（风险形成）、Trough→R3（趋势修复）、D1→P1（价格修复）。
- [ ] 展示MA20/MA25的Median/P75/P90/P95/Max完整统计并解释厚尾。
- [ ] 解释“见底快/趋势修复慢/价格修复更厚尾”的差异。
- [ ] 展示累计完成率与时长分桶，用文字解释20日、40日、60日的含义。
- [ ] 保留时长与跌幅相关性、最长事件表，并解释相关性不是因果。
- [ ] 明确P1与R1/R3是不同阈值，报告实际经验顺序而非强行规定理论顺序。

### Task 3: 重构条件风险分析

**Interfaces:**
- Produces `build_conditional_page(events) -> str`.

- [ ] 页面开头明确“条件风险”研究的是D3时点可知变量对未来甲3/乙/修复时间的区分能力。
- [ ] 首先给出MA20/25基准概率，作为所有条件比较的参照。
- [ ] 逐项解释D3偏离、D1→D3跌速、同期共振广度、MA60/120/200长期趋势、MA20→MA25升级、已持续未收回等状态。
- [ ] 每个条件模块必须写明：变量含义、数据是否呈单调关系、风险提升幅度、实战上能否在D3当时使用。
- [ ] 把“已持续未收回”明确标记为路径状态分析，而非D3初始预测变量。
- [ ] 末尾形成“从统计到风险判断框架”，总结基础风险→信号强度→市场共振→长期趋势→持续恶化的层层加码逻辑。

### Task 4: 页面顺序与正式报告清理

- [ ] 调整顶部导航与HTML页面物理顺序，使持续时间位于条件风险之前。
- [ ] 全局删除所有滚动/适配提示文案，但保留 `.table-scroll` CSS。
- [ ] 保持研究设计、时段、传导、事件明细内容不变。

### Task 5: 验证与审阅版留档

- [ ] 验证HTML仍包含8个page节点且各自唯一。
- [ ] 验证导航顺序与物理页面顺序一致。
- [ ] 验证HTML包含1505、研究设计更新文本、三日确认、风险形成/趋势修复/价格修复、条件风险判断框架。
- [ ] 验证不存在“← 横向滚动”“宽表保留完整研究字段”“请直接横向滚动”等提示。
- [ ] 验证 `.table-scroll` 仍包含 `overflow-x:auto`。
- [ ] 将审阅版HTML归档到 `ma_breakdown/review/v0.2/均线有效跌破统计_v0.2_三页叙事优化版.html`，不修改main。

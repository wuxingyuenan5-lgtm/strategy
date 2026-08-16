# 跌幅深度分析优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变V0.2事件定义、样本和原始数据的前提下，重构“跌幅深度分析”页，使其更清楚地展示厚尾风险、三日确认成本、Peak→D1→D3→Trough风险路径、指数风格差异、丙的后视镜修复结果与极端事件。

**Architecture:** 复用现有 `events` 与 `research` payload，不改事件引擎和数据抓取。新增一个独立的深度分析页面渲染模块，在最终HTML渲染阶段替换 `page-depth` 内容；同时统一删除报告中“横向滚动查看”等操作性辅助说明，但保留 `.table-scroll` 功能。

**Tech Stack:** Python 3.12, pandas/numpy（沿用现有依赖）, Jinja2, inline SVG, GitHub Actions。

## Global Constraints
- 不改变D1/D2/D3、R1/R2/R3、Peak、Trough、P1、甲1/甲2/甲3/乙/丙定义。
- 不改变1505个完成周期和12个进行中事件的研究口径。
- 不删除MA20/MA25完整宽表字段；横向滚动功能继续保留。
- 删除正式报告中“请横向滚动”“为适配屏幕”等开发/交互提示文字。
- 风险分位明确标记为“风险P75/P90/P95/P99”；风险P90表示约10%的历史事件更差。
- 标准差、概率、成功率等无方向比例不显示“+”号。

---

### Task 1: 深度分析派生统计与图表

**Files:**
- Create: `ma_breakdown/src/depth_report.py`
- Modify: `ma_breakdown/src/render.py`
- Test: `ma_breakdown/tests/test_depth_report.py`

**Interfaces:**
- Consumes: `payload["events"]`, `payload["research"]["depth_tables"]`, `payload["research"]["ma_summaries"]`, `payload["research"]["extreme_events"]`.
- Produces: `render_depth_page(payload: dict) -> str` and `replace_depth_page(html: str, payload: dict) -> str`.

- [ ] **Step 1:** 写测试，要求新版深度页包含“风险P90解释”“三日确认：确认收益与确认成本”“Peak→D1→D3→Trough”“R1价格修复结果（R3确认后回看）”“市场阶段”，并且不包含横向滚动提示文案。
- [ ] **Step 2:** 实现事件级派生统计：`peak_to_d1`、`d1_to_d3`、`a3`、`b`；按MA20/25统计中位数、风险P90/P95、阈值概率，以及 `a3-a1` 的事件级风险收敛。
- [ ] **Step 3:** 生成两张inline SVG：甲1/甲2/甲3风险分位带；各指数乙的Median/P90/P95比较。
- [ ] **Step 4:** 重构页面层级并保留MA20/MA25完整宽表。
- [ ] **Step 5:** 运行相关测试并提交。

### Task 2: 报告清理与格式统一

**Files:**
- Modify: `ma_breakdown/src/render.py`
- Test: `ma_breakdown/tests/test_depth_report.py`

**Interfaces:**
- Consumes: 最终rendered HTML。
- Produces: 不包含 `.scroll-note` 节点和横向滚动说明的正式报告。

- [ ] **Step 1:** 在最终渲染阶段移除所有 `<p class="scroll-note">...</p>` 及已知辅助性说明。
- [ ] **Step 2:** 深度页中无方向比例统一使用普通百分比格式，保留方向收益使用正负号。
- [ ] **Step 3:** 验证 `.table-scroll` CSS仍为 `overflow-x:auto`。
- [ ] **Step 4:** 提交清理。

### Task 3: 生成审阅版HTML

**Files:**
- Generate: `ma_breakdown/output/v0.2/index.html`

**Interfaces:**
- Consumes: 当前V0.2已冻结数据与现有events/research payload。
- Produces: 用户审阅用HTML。

- [ ] **Step 1:** 使用当前冻结数据渲染报告，不改变事件样本。
- [ ] **Step 2:** 校验8个子页、1505完成周期、研究设计页不回退、深度页新模块存在。
- [ ] **Step 3:** 将最终HTML复制为用户下载文件并交付。

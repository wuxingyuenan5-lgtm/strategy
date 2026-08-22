# A股成交额集中度因子研究

## 目标

研究指标：

> 前5%个股成交额占全市场成交额比例

定义：

`congestion = top5_amount / total_market_amount`

## 研究流程

1. AKShare 获取公开历史序列（用于快速验证）
2. 聚宽研究环境获取指数行情和进一步复算
3. 分析：
   - 因子分布
   - 滚动分位数
   - 未来1/3/5/10/20日收益
   - 高拥挤事件表现
   - 大小盘风格表现

## 当前范围

V0阶段只研究单因子，不扩展市场状态模型。

## 文件

- `akshare_download.py`
  - 下载AKShare现成拥挤度序列

- `joinquant_analysis.py`
  - 聚宽研究环境执行脚本（后续上传）

- `research_notes.md`
  - 研究记录

from __future__ import annotations

import re
from html import escape
from typing import Any

import numpy as np
import pandas as pd


_RELEASE_CSS = r"""
/* v0.2 approved overview + duration release layer */
.formal-kicker{font-size:12px;letter-spacing:.04em;color:#86868b;margin-bottom:8px}
.metric-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:20px 0}
.metric-box{background:#fafafa;border:1px solid #ececef;border-radius:10px;padding:15px 16px}
.metric-box .v{font-size:21px;font-weight:700;line-height:1.25}.metric-box .k{font-size:11.5px;color:#777;margin-top:5px;line-height:1.5}
.takeaway-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}
.takeaway{border:1px solid #e4e4e7;border-radius:10px;padding:16px;background:#fff}.takeaway .t{font-size:12px;color:#777;margin-bottom:4px}.takeaway .v{font-size:18px;font-weight:700;margin-bottom:5px}.takeaway .d{font-size:12.5px;color:#555;line-height:1.65}
.flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:10px;margin:20px 0}.flow .node{background:#fafafa;border:1px solid #e5e5e8;border-radius:10px;padding:14px;min-height:92px}.flow .node b{display:block;font-size:14px;margin-bottom:4px}.flow .node span{display:block;font-size:12px;color:#666;line-height:1.55}.flow .arr{color:#9a9aa0;font-size:22px}
.conclusion-list{counter-reset:item;margin-top:8px}.conclusion{position:relative;padding-left:34px;margin:16px 0;font-size:14.5px;line-height:1.8}.conclusion:before{counter-increment:item;content:counter(item);position:absolute;left:0;top:1px;width:22px;height:22px;border-radius:50%;background:#1d1d1f;color:#fff;font-size:11px;display:flex;align-items:center;justify-content:center}
.threshold{display:grid;grid-template-columns:110px 1fr 1fr;gap:0;border:1px solid #e7e7ea;border-radius:10px;overflow:hidden;margin:16px 0}.threshold>div{padding:13px 14px;border-bottom:1px solid #ececef}.threshold>div:nth-last-child(-n+3){border-bottom:0}.threshold .head{background:#fafafa;font-size:11.5px;color:#777;font-weight:600}.threshold .stage{font-weight:700;background:#fcfcfd}.threshold .risk{font-size:13px}.threshold .meaning{font-size:13px;color:#555}
.report-details{border:1px solid #e5e5e8;border-radius:10px;margin:22px 0;background:#fff}.report-details>summary{cursor:pointer;padding:14px 16px;font-weight:600;list-style:none}.report-details>summary::-webkit-details-marker{display:none}.report-details>summary:after{content:'展开';float:right;font-size:11px;color:#888;font-weight:400}.report-details[open]>summary:after{content:'收起'}.report-details .details-body{padding:0 16px 16px}
@media(max-width:800px){.metric-grid,.takeaway-grid{grid-template-columns:1fr 1fr}.flow{grid-template-columns:1fr}.flow .arr{transform:rotate(90deg);text-align:center}.threshold{grid-template-columns:1fr}.threshold>div{border-bottom:1px solid #ececef!important}.threshold>div:last-child{border-bottom:0!important}}
@media(max-width:560px){.metric-grid,.takeaway-grid{grid-template-columns:1fr}}
"""


def _pct(v: Any, digits: int = 1, sign: bool = False) -> str:
    try:
        if v is None or pd.isna(v):
            return "—"
        x=float(v)*100
        return f"{x:+.{digits}f}%" if sign else f"{x:.{digits}f}%"
    except Exception:
        return "—"


def _num(v: Any, digits: int = 0) -> str:
    try:
        if v is None or pd.isna(v):
            return "—"
        return f"{float(v):.{digits}f}"
    except Exception:
        return "—"


def _stats(s: pd.Series) -> dict[str, float | int | None]:
    x=pd.to_numeric(s,errors="coerce").dropna()
    if x.empty:
        return {"n":0,"median":None,"p75":None,"p90":None,"p95":None,"mean":None,"max":None}
    return {
        "n":int(len(x)),"median":float(x.median()),"p75":float(x.quantile(.75)),
        "p90":float(x.quantile(.90)),"p95":float(x.quantile(.95)),"mean":float(x.mean()),"max":float(x.max())
    }


def _replace_page(html: str, page_id: str, replacement: str) -> str:
    start=html.find(f'<div class="page" id="page-{page_id}">')
    if start < 0:
        raise ValueError(f"page not found: {page_id}")
    nxt=html.find('<div class="page" id="page-', start+1)
    end=nxt if nxt >= 0 else html.find('</main>', start)
    if end < 0:
        raise ValueError(f"page end not found: {page_id}")
    return html[:start]+replacement+"\n"+html[end:]


def _ma_rows(ev: pd.DataFrame) -> dict[int, dict[str, dict[str, float | int | None]]]:
    out={}
    for ma in (20,25):
        g=ev[ev["ma_window"]==ma]
        out[ma]={k:_stats(g[k]) for k in ("d1_to_trough","trough_to_r1","trough_to_r3","d1_to_r3","d1_to_p1")}
    return out


def _p1_timing(ev: pd.DataFrame) -> dict[int, dict[str, Any]]:
    out={}
    for ma in (20,25):
        g=ev[ev["ma_window"]==ma].copy()
        hit=g[pd.to_numeric(g.get("p1_idx"),errors="coerce").notna()].copy()
        n=len(hit)
        if n:
            p1=pd.to_numeric(hit["p1_idx"],errors="coerce");r1=pd.to_numeric(hit["r1_idx"],errors="coerce");r3=pd.to_numeric(hit["r3_idx"],errors="coerce")
            before=float((p1<r1).mean());equal=float((p1==r1).mean());after=float((p1>r1).mean());le_r3=float((p1<=r3).mean());gt_r3=float((p1>r3).mean())
        else:
            before=equal=after=le_r3=gt_r3=np.nan
        out[ma]={"hits":n,"before_r1":before,"equal_r1":equal,"after_r1":after,"le_r3":le_r3,"gt_r3":gt_r3,"censored":int(len(g)-n)}
    return out


def _recovery(ev: pd.DataFrame, horizons=(3,5,10,20,40,60,120)) -> list[dict[str, Any]]:
    n=len(ev)
    rows=[]
    for h in horizons:
        rows.append({
            "days":h,
            "trough":float((pd.to_numeric(ev["d1_to_trough"],errors="coerce")<=h).sum()/n) if n else 0.0,
            "r3":float((pd.to_numeric(ev["d1_to_r3"],errors="coerce")<=h).sum()/n) if n else 0.0,
            "p1":float((pd.to_numeric(ev["d1_to_p1"],errors="coerce")<=h).sum()/n) if n else 0.0,
        })
    return rows


def _unrecovered(ev: pd.DataFrame) -> list[dict[str, Any]]:
    rows=[]
    for ma in (20,25):
        base=ev[ev["ma_window"]==ma].copy();dur=pd.to_numeric(base["d1_to_r3"],errors="coerce")
        for elapsed in (5,10,20,40):
            g=base[dur>elapsed].copy()
            if g.empty:
                continue
            rem=pd.to_numeric(g["d1_to_r3"],errors="coerce")-elapsed
            a3=pd.to_numeric(g["a3"],errors="coerce");b=pd.to_numeric(g["b"],errors="coerce")
            rows.append({"ma":ma,"elapsed":elapsed,"n":len(g),"next5":float((rem<=5).mean()),"remaining":float(rem.median()),"a3":float(a3.median()),"p10":float((a3<=-.10).mean()),"b":float(b.median())})
    return rows


def _overview_html(payload: dict, ev: pd.DataFrame, ma: dict[int, dict[str, dict[str, Any]]], p1: dict[int, dict[str, Any]]) -> str:
    meta=payload.get("meta",{});overview=payload.get("overview",{})
    med_r3=float(pd.to_numeric(ev["d1_to_r3"],errors="coerce").median()) if len(ev) else np.nan
    hit_total=p1[20]["hits"]+p1[25]["hits"]
    after_total=(p1[20]["gt_r3"]*p1[20]["hits"]+p1[25]["gt_r3"]*p1[25]["hits"])/hit_total if hit_total else np.nan
    p90_20=ma[20]["d1_to_p1"]["p90"];p90_25=ma[25]["d1_to_p1"]["p90"]
    return f'''<div class="page" id="page-overview"><div class="section"><div class="card">
<div class="formal-kicker">V0.2 正式研究版 · P1=D0价格锚 · 描述性历史研究</div>
<h2>研究概览</h2>
<p>本报告研究7个A股指数的MA20 / MA25有效跌破，从D1候选、D3真跌破确认，到R3趋势修复确认的完整均线信号周期。冻结样本为 <b>{meta.get('event_count','—')}</b> 个完成周期（MA20 804、MA25 701）和{meta.get('ongoing_count','—')}个进行中真跌破事件，数据截至{escape(str(meta.get('cutoff','')))}。</p>
<div class="hl"><b>一句话结论：</b>均线跌破的“典型事件”通常是浅回撤、两到四周完成趋势修复；真正需要防范的是少数厚尾事件——它们不仅跌得更深，而且会长期无法R3，甚至在趋势重新站稳均线后，绝对价格仍可能数月乃至更久无法收复跌破前D0水平。</div>
<div class="metric-grid">
<div class="metric-box"><div class="v">{_pct(overview.get('day1_success_rate'))}</div><div class="k">D1候选最终形成D3真跌破<br>D1本身仍包含较多噪音</div></div>
<div class="metric-box"><div class="v">{_pct(overview.get('median_a1'),sign=True)}</div><div class="k">甲1中位<br>典型事件的最大回撤并不深</div></div>
<div class="metric-box"><div class="v">{_pct(overview.get('p90_a1'),sign=True)}</div><div class="k">甲1风险P90<br>约10%的事件比这一水平更差</div></div>
<div class="metric-box"><div class="v">≈{_num(med_r3)}日</div><div class="k">D1→R3总体中位<br>典型均线信号周期约2–4周</div></div>
<div class="metric-box"><div class="v">≈{_pct(after_total,0)}</div><div class="k">有P1样本中，P1发生在R3之后<br>趋势修复不等于价格修复</div></div>
<div class="metric-box"><div class="v">{_num(p90_20)} / {_num(p90_25)}日</div><div class="k">MA20 / MA25 的D1→P1 P90<br>价格修复存在极长右尾</div></div>
</div>
<h3>六条核心结论</h3><div class="conclusion-list">
<div class="conclusion"><b>真跌破不是“大跌”的同义词。</b> D1候选中约{_pct(overview.get('day1_success_rate'))}最终走到D3；而进入真跌破样本以后，甲1中位约{_pct(overview.get('median_a1'),sign=True)}。因此均线信号首先是趋势状态变化，而不是对跌幅大小的直接判断。</div>
<div class="conclusion"><b>风险的核心不是均值，而是厚尾。</b> 典型事件回撤有限，但甲1风险P90约{_pct(overview.get('p90_a1'),sign=True)}。少数系统性行情显著拉长跌幅和持续时间，研究重点应放在“如何识别尾部状态”，而不是只记住平均跌幅。</div>
<div class="conclusion"><b>D3确认提高信号可靠性，但并不意味着风险已经释放完。</b> 等待三日确认会过滤一部分假跌破，也会消耗部分下跌空间；确认后仍需结合D3负乖离、短期跌速和长期趋势状态判断剩余尾部风险。</div>
<div class="conclusion"><b>时间本身是重要的动态状态变量。</b> 普通周期大多在约两到四周内完成R3；但若20日甚至40日仍未R3，历史上存活到这一状态的事件会明显向深跌、长周期样本集中。这里反映的是条件样本结构的变化，不是“时间越久必然继续下跌”的因果关系。</div>
<div class="conclusion"><b>趋势修复和价格修复必须分开。</b> R3只表示价格连续3日重新站上移动平均线；P1要求重新收复D1前一交易日D0的收盘价。有P1样本中约{_pct(after_total)}的P1晚于R3，而且P1的P90超过400个交易日，说明“均线恢复”可以远早于“原价格水平恢复”。</div>
<div class="conclusion"><b>市场阶段与系统性共振决定同一个信号的含义。</b> 2008更接近“深跌+慢修复”，2011–2014偏“慢性拖延”，2015体现“中位不深、尾部极深”，2019–2021更接近“浅跌+快修复”；跨指数层面则以同步共振为主，固定领先链条较弱。</div>
</div>
<h3>阅读逻辑</h3><div class="table-scroll"><table class="reading-map"><thead><tr><th>问题</th><th>应看什么</th><th>最终要得到的判断</th></tr></thead><tbody>
<tr><td>跌破后会跌多深？</td><td>跌幅深度分析</td><td>先看典型回撤，再看P90/P95尾部；不要用均值代表风险</td></tr>
<tr><td>风险多久形成、多久修复？</td><td>持续时间分析</td><td>区分Trough、R3和P1三个不同的时间终点</td></tr>
<tr><td>D3确认后什么状态更危险？</td><td>条件风险分析</td><td>用当时可知的负乖离、跌速、长期趋势做风险分层</td></tr>
<tr><td>不同历史行情能否直接类比？</td><td>时段与信号图表</td><td>先定位市场阶段，再解释同一均线信号</td></tr>
<tr><td>是否存在稳定的指数领先关系？</td><td>跨指数传导</td><td>优先识别共振与扩散，不机械构造固定领先链</td></tr>
</tbody></table></div>
<div class="warn"><b>方法边界：</b>主统计以已完成R3的均线信号周期为对象。Trough、甲1/2/3、乙、丙以及部分时间路径指标具有事后属性；“已持续X日仍未R3”的统计是条件于历史存活状态的结果分布，不能直接解释为因果预测。</div>
</div></div></div>'''


def _duration_html(payload: dict, ev: pd.DataFrame, ma: dict[int, dict[str, dict[str, Any]]], p1: dict[int, dict[str, Any]]) -> str:
    recovery=_recovery(ev);unrec=_unrecovered(ev)
    rows=[]
    labels=[("d1_to_trough","D1→Trough：风险形成"),("trough_to_r3","Trough→R3：趋势修复"),("d1_to_r3","D1→R3：完整信号周期"),("d1_to_p1","D1→P1：首次收复D0价格")]
    for m in (20,25):
        for key,label in labels:
            s=ma[m][key]
            rows.append(f'<tr><td>MA{m}</td><td>{label}</td><td>{s["n"]}</td><td>{_num(s["median"])}</td><td>{_num(s["p75"])}</td><td>{_num(s["p90"])}</td><td>{_num(s["p95"])}</td><td>{_num(s["mean"],1)}</td><td>{_num(s["max"])}</td></tr>')
    stage_rows=[]
    for m in (20,25):
        for key,label in (("trough_to_r1","Trough→R1"),("trough_to_r3","Trough→R3")):
            s=ma[m][key]
            stage_rows.append(f'<tr><td>MA{m}</td><td>{label}</td><td>{s["n"]}</td><td>{_num(s["median"])}</td><td>{_num(s["p75"])}</td><td>{_num(s["p90"])}</td><td>{_num(s["mean"],1)}</td><td>{_num(s["max"])}</td></tr>')
    p1_rows=[]
    for m in (20,25):
        x=p1[m];p1_rows.append(f'<tr><td>MA{m}</td><td>{x["hits"]}</td><td>{_pct(x["before_r1"])}</td><td>{_pct(x["equal_r1"])}</td><td>{_pct(x["after_r1"])}</td><td>{_pct(x["le_r3"])}</td><td>{_pct(x["gt_r3"])}</td><td>{x["censored"]}</td></tr>')
    recovery_rows=''.join(f'<tr><td>{r["days"]}日</td><td>{_pct(r["trough"])}</td><td>{_pct(r["r3"])}</td><td>{_pct(r["p1"])}</td></tr>' for r in recovery)
    unrec_rows=''.join(f'<tr><td>MA{r["ma"]}</td><td>已运行{r["elapsed"]}日仍未R3</td><td>{r["n"]}</td><td>{_pct(r["next5"])}</td><td>{_num(r["remaining"],1)}</td><td>{_pct(r["a3"],sign=True)}</td><td>{_pct(r["p10"])}</td><td>{_pct(r["b"],sign=True)}</td></tr>' for r in unrec)
    support=[]
    research=payload.get("research",{})
    corr=research.get("duration_correlations",[])
    if corr:
        support.append('<h4>持续时间与跌幅相关性底稿</h4><div class="table-scroll"><table><thead><tr><th>指数</th><th>MA</th><th>N</th><th>D1→R3 vs |甲1|</th><th>D1→R3 vs |甲3|</th><th>D1→R3 vs |乙|</th></tr></thead><tbody>'+''.join(f'<tr><td>{escape(str(x.get("index_name","")))}</td><td>MA{x.get("ma_window","")}</td><td>{x.get("n","")}</td><td>{_num(x.get("corr_a1"),2)}</td><td>{_num(x.get("corr_a3"),2)}</td><td>{_num(x.get("corr_b"),2)}</td></tr>' for x in corr)+'</tbody></table></div>')
    charts=payload.get("charts",{})
    if charts.get("duration_svg"):
        support.append('<h4>累计完成率曲线</h4><div class="chart">'+str(charts["duration_svg"]).replace('已收复D1价格','已收复D0价格')+'</div>')
    if charts.get("window_svg"):
        support.append('<h4>固定窗口风险路径</h4><div class="chart">'+str(charts["window_svg"])+'</div>')
    details=''.join(support)
    return f'''<div class="page" id="page-duration"><div class="section">
<div class="card"><div class="formal-kicker">核心问题：风险形成、趋势修复、价格修复分别需要多久？</div><h2>持续时间分析：时间不是一个数字，而是三种不同的风险时钟</h2>
<p>跌幅深度回答“会跌多深”，持续时间分析则回答“风险在哪个阶段、还可能拖多久”。这里必须把三个终点拆开：<b>D1→Trough</b>衡量最终低点何时形成；<b>Trough→R3 / D1→R3</b>衡量趋势重新站稳均线需要多久；<b>D1→P1</b>衡量跌破前D0绝对价格何时真正被重新收复。</p>
<div class="hl"><b>先看结论：</b>典型事件的风险释放和趋势修复都不算慢——最终Trough中位约7日，见底后再约7–8日完成R3；真正的长尾来自两处：一是少数事件迟迟不形成最终低点，二是即使R3已经完成，绝对价格仍可能长期无法收复D0。因此，“还没R3多久”与“R3后是否已收复D0”是两个不同的风险问题。</div>
<div class="flow"><div class="node"><b>风险形成</b><span>D1 → Trough<br>中位约7日<br>尾部可持续30–50日以上</span></div><div class="arr">→</div><div class="node"><b>趋势修复</b><span>Trough → R3<br>中位约7–8日<br>P90约13–16日</span></div><div class="arr">→</div><div class="node"><b>价格修复</b><span>D1 → P1（收复D0）<br>中位19–22日<br>P90高达436–452日</span></div></div>
<h3>1. 三个时间尺度：中位数接近，但尾部完全不同</h3><div class="table-scroll"><table><thead><tr><th>均线</th><th>时间尺度</th><th>N</th><th>中位</th><th>P75</th><th>P90</th><th>P95</th><th>均值</th><th>最长</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<div class="takeaway-grid"><div class="takeaway"><div class="t">典型风险形成</div><div class="v">约7个交易日</div><div class="d">普通事件的最终低点往往在D1后一到两周内形成，但P90已拉长到30–35日。</div></div><div class="takeaway"><div class="t">见底后的趋势确认</div><div class="v">再等7–8日</div><div class="d">Trough出现以后，R3的时间分布明显更集中，说明“见底后站稳均线”通常比“等待最终低点”更可控。</div></div><div class="takeaway"><div class="t">真正的时间长尾</div><div class="v">P1 P90 &gt; 400日</div><div class="d">P1中位只比R3略晚，但上分位急剧拉长：典型样本不慢，少数长期套牢样本极慢。</div></div></div>
<p><b>这里最值得强调的不是“中位数差几天”，而是分布形状不同。</b> D1→R3的P90只有38–45日，而D1→P1的P90直接扩大到436–452日。也就是说，大多数事件的趋势和价格都会较快修复，但一旦进入价格修复的右尾，等待时间会从“几周”跃迁到“数月甚至数年”。</p></div>
<div class="card"><h3>2. 风险形成与趋势修复并不对称：长周期首先是“迟迟不见底”</h3><div class="table-scroll"><table><thead><tr><th>均线</th><th>阶段</th><th>N</th><th>中位</th><th>P75</th><th>P90</th><th>均值</th><th>最长</th></tr></thead><tbody>{''.join(stage_rows)}</tbody></table></div><p>最终Trough出现以后，典型事件约5–6日进入最终成功收回序列R1，再经过两日确认到R3。更重要的是，<b>Trough→R3的尾部明显短于D1→Trough</b>。</p><div class="subtle"><b>因此，从历史分布结构看，长周期事件的主要问题通常不是“已经见底但迟迟站不回均线”，而是“最终低点本身迟迟没有形成”。</b> 这不是说R3一定容易，而是说明持续未R3的事件中，首先需要警惕的是风险仍在寻找最终Trough。</div></div>
<div class="card"><h3>3. R3不是“解套”：价格修复明显慢于趋势修复</h3><p><b>D0是D1前一个交易日。</b>正式P1定义为D1以后第一次Close ≥ D0 Close，代表重新回到均线跌破发生前的绝对价格锚点。因此R3回答“趋势是否站稳”，P1回答“价格是否真正回来了”。</p><div class="table-scroll"><table><thead><tr><th>均线</th><th>有P1样本</th><th>P1早于R1</th><th>P1=R1</th><th>P1晚于R1</th><th>P1不晚于R3</th><th>P1晚于R3</th><th>截至样本末仍无P1</th></tr></thead><tbody>{''.join(p1_rows)}</tbody></table></div><div class="hl"><b>这个关系现在非常清楚：</b>约72%的P1晚于R1，约56%的P1晚于R3。也就是说，市场重新站稳MA20/MA25以后，价格仍可能低于跌破发生前的D0水平。R3应被理解为“趋势状态改善”，而不是“原价格风险已经修复”。</div><p>P1并不被定义为必须晚于R1或R3。移动平均线本身会随行情变化，因此少数事件可以先收复D0价格、但当日仍未满足最终成功R1/R3条件。本报告保留真实先后关系，不人为约束。</p></div>
<div class="card"><h3>4. 累计完成率：20日以后，趋势修复和价格修复开始明显分叉</h3><div class="table-scroll"><table><thead><tr><th>D1后时间</th><th>已到最终Trough</th><th>已完成R3</th><th>已收复D0价格P1</th></tr></thead><tbody>{recovery_rows}</tbody></table></div><p>前20日三个过程都在推进，但此后差异迅速扩大：40日时大部分事件已完成R3，而价格修复明显滞后；120日时均线信号周期本身已基本结束，仍有相当一部分样本没有重新触及跌破前D0收盘价。</p><div class="subtle"><b>持续时间分析真正要告诉读者的，是“趋势修复”与“价格修复”存在结构性时间差，而不是只给一个平均天数。</b></div></div>
<div class="card"><h3>5. 已持续未R3：时间如何动态更新风险判断？</h3><p>D3只是初始确认时点。事件继续运行以后，“已经持续X日仍未R3”会成为新的路径信息。这里最容易误读：<b>持续更久并不意味着未来5日内完成R3的概率会单调下降</b>；真正稳定的信号是，能存活到更久仍未R3的历史样本，其最终风险分布越来越差。</p><div class="table-scroll"><table><thead><tr><th>均线</th><th>当前状态</th><th>仍在运行样本</th><th>未来5日内完成R3</th><th>剩余R3时长中位</th><th>甲3中位</th><th>P(甲3≤-10%)</th><th>乙中位</th></tr></thead><tbody>{unrec_rows}</tbody></table></div>
<div class="threshold"><div class="head">持续状态</div><div class="head">历史最终风险分布</div><div class="head">应该怎样理解</div><div class="stage">10日仍未R3</div><div class="risk">已经脱离快速修复样本，严重事件占比开始抬升</div><div class="meaning">不能仅凭时间判断下一周一定继续恶化。</div><div class="stage">20日仍未R3</div><div class="risk">甲3中位约-8%左右，严重事件占比约四成</div><div class="meaning"><b>最有实务意义的风险升级点。</b> 历史样本明显更容易演化为深跌周期。</div><div class="stage">40日仍未R3</div><div class="risk">甲3中位约-14%左右，严重事件占比约八成</div><div class="meaning"><b>极端长周期状态。</b> 样本已高度集中于历史尾部事件。</div></div>
<p><b>因此，持续时间最有价值的用法不是“预测还要几天反弹”，而是动态重估这次事件属于普通样本还是尾部样本。</b> 5–10日未R3仍较常见；20日未R3时风险分布已经明显恶化；40日未R3则几乎进入极端事件集合。与此同时，未来5日完成R3的概率并没有随持续时间单调下降，所以不能把这些阈值机械地解释成短期择时信号。</p><div class="warn"><b>重要解释边界：</b>表中的甲3和乙是事件最终完成后的结果变量。正确读法是“历史上已经持续到X日仍未R3的事件，最终结果通常如何”，而不是“到了第X日就已经知道未来一定会跌到这些水平”。这是条件分布更新，不是因果预测。</div></div>
<div class="card"><h3>6. 最终应该从持续时间页带走什么？</h3><div class="conclusion-list"><div class="conclusion"><b>普通事件：</b>最终Trough大约一周形成，见底后再约一周完成R3，完整信号周期通常落在两到四周。</div><div class="conclusion"><b>长周期事件：</b>真正值得警惕的是“迟迟不形成最终低点”。D1→Trough的尾部显著长于Trough→R3。</div><div class="conclusion"><b>动态风险：</b>20日仍未R3后，历史最终风险分布已经明显恶化；40日仍未R3基本属于极端尾部状态。但这不是短期反弹倒计时。</div><div class="conclusion"><b>价格风险：</b>R3只是趋势修复。约56%的P1发生在R3之后，P1的P90超过400个交易日；所以“站回均线”与“收复跌破前价格”必须分开管理。</div><div class="conclusion"><b>MA20与MA25：</b>MA25整体稍慢，但两条均线呈现的是同一套时间结构。均线长度改变速度，不改变“风险形成—趋势修复—价格修复”的核心结论。</div></div><div class="hl"><b>压缩成一句话：</b>均线跌破的时间风险不是“跌破后平均多久收回”，而是——普通事件通常两到四周结束；如果20–40日仍未R3，应把它从普通技术调整重新分类为潜在尾部事件；即使R3已经出现，也不能把它等同于原价格风险已经修复。</div>
<details class="report-details"><summary>研究底稿：累计曲线、相关性与固定窗口</summary><div class="details-body">{details}</div></details></div>
</div></div>'''


def apply_v02_release_patch(html: str, payload: dict) -> str:
    """Apply the user-approved V0.2 overview/duration narrative and D0-based P1 terminology.

    The event detector remains unchanged. This layer only consumes the already-rendered
    payload, derives descriptive duration summaries, and replaces the two approved pages.
    """
    ev=pd.DataFrame(payload.get("events",[]))
    required={"ma_window","d1_to_trough","trough_to_r1","trough_to_r3","d1_to_r3","d1_to_p1","p1_idx","r1_idx","r3_idx","a3","b"}
    missing=sorted(required-set(ev.columns))
    if missing:
        raise ValueError("release patch missing event fields: "+", ".join(missing))
    ma=_ma_rows(ev);p1=_p1_timing(ev)
    if "v0.2 approved overview + duration release layer" not in html:
        html=html.replace("</style>",_RELEASE_CSS+"\n</style>",1)
    html=re.sub(
        r'<a href="#depth" data-page="depth">跌幅深度分析</a><a href="#conditional" data-page="conditional">条件风险分析</a><a href="#duration" data-page="duration">持续时间分析</a>',
        '<a href="#depth" data-page="depth">跌幅深度分析</a><a href="#duration" data-page="duration">持续时间分析</a><a href="#conditional" data-page="conditional">条件风险分析</a>',
        html,
        count=1,
    )
    html=_replace_page(html,"overview",_overview_html(payload,ev,ma,p1))
    html=_replace_page(html,"duration",_duration_html(payload,ev,ma,p1))
    html=html.replace("<b>P1：</b>Day1以后第一次Close≥D1 Close，不以R3作为搜索终点。", "<b>D0：</b>D1前一交易日。<b>P1：</b>D1以后第一次Close≥D0 Close，搜索持续到样本截止；未达到则记为右删失，不人为约束P1必须晚于R1/R3。")
    html=html.replace("已收复D1价格","已收复D0价格").replace("收复D1价格","收复D0价格").replace("D1价格首次收复","D0价格首次收复")
    html=html.replace("D1→价格收复P1","D1→D0价格收复P1")
    html=html.replace("D1→收复D1价格","D1→收复D0价格")
    html=html.replace("<p>P1定义为", "<p><b>D1→收复D0价格：</b>P1定义为")
    html=html.replace("</main>","<!-- V02_RELEASE_CONTRACT: P1_D0 + APPROVED_OVERVIEW_DURATION -->\n</main>",1)
    return html

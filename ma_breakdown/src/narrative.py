from __future__ import annotations

def _pct(v,d=1): return "—" if v is None else f"{v*100:.{d}f}%"
def _num(v,d=1): return "—" if v is None else f"{v:.{d}f}"

def build_narrative(research:dict,conditional_panels:list[dict])->dict:
    sig=research["signal_reliability"][0];mas={x["ma_window"]:x for x in research["ma_summaries"]};m20=mas.get(20,{});m25=mas.get(25,{})
    idx_rows=[]
    for r in research["style_comparison"]:
        vals=[r[k]["b_mean"] for k in ("ma20","ma25") if k in r]
        if vals:idx_rows.append((sum(vals)/len(vals),r["index_name"]))
    idx_rows.sort();deepest=idx_rows[0] if idx_rows else (None,"—");shallowest=idx_rows[-1] if idx_rows else (None,"—")
    ov=[f"1. <b>三日确认显著过滤噪声，但不是低成本确认。</b> Day1候选最终形成Day3真跌破的比例为 {_pct(sig['day1_success_rate'])}；已经走到Day2后，条件成功率升至 {_pct(sig['day2_success_rate'])}。这意味着首日跌破中约 {_pct(sig['fake_rate'])} 最终不会完成三日真跌破。"]
    if m20 and m25:
        ov += [
        f"2. <b>真跌破后的下行仍呈明显厚尾。</b> MA20下甲1中位为 {_pct(m20['a1']['median'])}、P90为 {_pct(m20['a1']['p90_risk'])}；MA25分别为 {_pct(m25['a1']['median'])} 和 {_pct(m25['a1']['p90_risk'])}。因此单看平均值仍会低估少数深跌事件。",
        f"3. <b>等待Day3确认会减少剩余下跌，但不会消除尾部风险。</b> MA20甲1/甲3中位从 {_pct(m20['a1']['median'])} 收敛至 {_pct(m20['a3']['median'])}，MA25从 {_pct(m25['a1']['median'])} 收敛至 {_pct(m25['a3']['median'])}；确认后的风险仍具有显著长尾。",
        f"4. <b>乙揭示的整轮回撤明显深于单纯从D1往后看的甲。</b> MA20乙中位为 {_pct(m20['b']['median'])}，MA25为 {_pct(m25['b']['median'])}，说明不少事件在Day1出现以前已经从近10日高点完成了一段回撤。",
        f"5. <b>MA25依旧更慢、事件更少，同时对应的完整回撤更深。</b> 完整事件数MA20为 {m20['n']} 次、MA25为 {m25['n']} 次；乙均值分别为 {_pct(m20['b']['mean'])} 和 {_pct(m25['b']['mean'])}，D1→R3中位时间分别为 {_num(m20['d1_to_r3']['median'])} 与 {_num(m25['d1_to_r3']['median'])} 个交易日。"]
    if deepest[0] is not None:ov.append(f"6. <b>指数风格差异仍然明显。</b> 按MA20/25的乙平均值综合观察，{deepest[1]}的完整回撤最深（约 {_pct(deepest[0])}），{shallowest[1]}相对最浅（约 {_pct(shallowest[0])}）。这说明同样的均线跌破在不同风格指数上不能使用同一风险尺度。")
    ps=research["period_summary"]
    if ps:
        worst=min(ps,key=lambda x:x["a1"]["mean"]);best=max(ps,key=lambda x:x["a1"]["mean"])
        ov.append(f"7. <b>市场阶段仍是核心解释变量。</b> {worst['period']}的甲1平均跌幅达到 {_pct(worst['a1']['mean'])}，而{best['period']}约为 {_pct(best['a1']['mean'])}；同一个技术信号在不同市场环境下的风险尺度差异远大于单一均线参数差异。")
    ct=research.get("conduction_top",[])
    if ct:
        top=max(ct,key=lambda x:x["probability"]);ov.append(f"8. <b>跨指数风险表现为高相关链条而非唯一领先指标。</b> 在样本数不少于20次的组合中，MA{top['ma_window']}下“{top['source_name']}→{top['target_name']}”在{top['horizon']}个交易日窗口内的条件触发率最高，约 {_pct(top['probability'])}；更适合把跨指数信号理解为共振确认。")
    depth=[]
    if m20 and m25:
        depth=[
        f"全样本显示，甲1→甲2→甲3的中位风险逐级收敛：MA20为 {_pct(m20['a1']['median'])} → {_pct(m20['a2']['median'])} → {_pct(m20['a3']['median'])}；MA25为 {_pct(m25['a1']['median'])} → {_pct(m25['a2']['median'])} → {_pct(m25['a3']['median'])}。这量化了等待确认带来的剩余风险减少，但P90仍然处于较深区间。",
        f"乙的风险尺度明显高于甲1：MA20乙中位 {_pct(m20['b']['median'])}、P90 {_pct(m20['b']['p90_risk'])}；MA25乙中位 {_pct(m25['b']['median'])}、P90 {_pct(m25['b']['p90_risk'])}。乙回答整轮回撤有多深，甲1/2/3回答看到不同确认阶段以后还剩多少风险。",
        f"丙依然呈现中位接近盈亏平衡、均值明显更差的特征：MA20中位 {_pct(m20['c']['median'])}、均值 {_pct(m20['c']['mean'])}；MA25中位 {_pct(m25['c']['median'])}、均值 {_pct(m25['c']['mean'])}。少数深跌周期会显著拖累均值。"]
    conditional=[]
    for p in conditional_panels:
        bs=p.get("buckets",[])
        if len(bs)>=2:
            lo,hi=bs[0],bs[-1];conditional.append({"feature":p.get("feature_label",p.get("feature","条件变量")),"text":f"从最低分组到最高分组，甲3中位由 {_pct(lo.get('median_a3'))} 变化至 {_pct(hi.get('median_a3'))}，跌超10%的概率由 {_pct(lo.get('prob_gt_10'))} 变化至 {_pct(hi.get('prob_gt_10'))}。{'该变量呈单调分层，可作为风险等级变量。' if p.get('monotonic') else '该变量存在区分度，但并非严格单调，实盘使用应避免机械阈值。'}"})
    dur=[]
    if m20 and m25:dur.append(f"均线真收回明显慢于最终见底。MA20 D1→R3中位约 {_num(m20['d1_to_r3']['median'])} 个交易日，MA25约 {_num(m25['d1_to_r3']['median'])} 个交易日；风险释放和趋势修复不是同一个时间点。")
    curves=research.get("recovery_curves",[]);h20=next((x for x in curves if x["days"]==20),None);h60=next((x for x in curves if x["days"]==60),None)
    if h20 and h60:dur.append(f"从累计完成率看，D1后20个交易日内已有 {_pct(h20['trough_rate'])} 的事件到达最终低点、{_pct(h20['r3_rate'])} 完成R3真收回，同期已有 {_pct(h20['p1_rate'])} 重新收复D1价格；到60日时三者分别为 {_pct(h60['trough_rate'])}、{_pct(h60['r3_rate'])}、{_pct(h60['p1_rate'])}。P1与R3不是嵌套关系，这正说明均线修复与价格修复是两条不同时间轴。")
    ws=research.get("window_summary",[]);ten=next((x for x in ws if x["days"]==10),None);twenty=next((x for x in ws if x["days"]==20),None)
    if ten and twenty:dur.append(f"固定窗口风险路径显示，D1后10日内最大跌幅中位为 {_pct(ten['median_maxdd'])}、P90为 {_pct(ten['p90_maxdd'])}；20日窗口扩大到 {_pct(twenty['median_maxdd'])} 和 {_pct(twenty['p90_maxdd'])}。这能判断风险主要在前10日释放，还是继续向20日以后扩散。")
    ptext=[];detail=research.get("period_detail",[])
    for p in ps:
        rows=[r for r in detail if r["period"]==p["period"]];extra=""
        if rows:
            deep=min(rows,key=lambda x:x["a1"]["mean"]);shallow=max(rows,key=lambda x:x["a1"]["mean"]);extra=f" 在该阶段的指数×均线组合中，{deep['index_name']} MA{deep['ma_window']}的甲1平均最深（{_pct(deep['a1']['mean'])}），{shallow['index_name']} MA{shallow['ma_window']}相对最浅（{_pct(shallow['a1']['mean'])}）。"
        ptext.append({"period":p["period"],"text":f"本阶段共 {p['n']} 个完整事件，甲1平均 {_pct(p['a1']['mean'])}、中位 {_pct(p['a1']['median'])}、P90 {_pct(p['a1']['p90_risk'])}；乙平均 {_pct(p['b']['mean'])}，丙中位 {_pct(p['c']['median'])}，D1→R3中位 {_num(p['duration']['median'])} 个交易日。{extra}"})
    ctext=[]
    for ma in (20,25):
        rows=[x for x in ct if x["ma_window"]==ma and x["horizon"]==10]
        if rows:
            top=max(rows,key=lambda x:x["probability"]);low=min(rows,key=lambda x:x["probability"]);ctext.append(f"MA{ma}的10日传导中，{top['source_name']}→{top['target_name']}条件触发率较高（{_pct(top['probability'])}，N={top['denominator']}）；样本内较弱的组合为{low['source_name']}→{low['target_name']}（{_pct(low['probability'])}，N={low['denominator']}）。跨指数关系更适合做风险共振确认，而不应直接解释为稳定因果领先。")
    return {"overview_conclusions":ov,"depth_commentary":depth,"conditional_commentary":conditional,"duration_commentary":dur,"period_commentary":ptext,"conduction_commentary":ctext}

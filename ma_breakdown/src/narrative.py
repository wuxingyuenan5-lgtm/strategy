from __future__ import annotations
import math

def _ok(v):
    try:return v is not None and not math.isnan(float(v))
    except Exception:return False

def _pct(v,d=1): return "—" if not _ok(v) else f"{float(v)*100:.{d}f}%"
def _num(v,d=1): return "—" if not _ok(v) else f"{float(v):.{d}f}"
def _avg(vals):
    x=[float(v) for v in vals if _ok(v)];return sum(x)/len(x) if x else None

def build_narrative(research:dict,conditional_panels:list[dict])->dict:
    sig=research["signal_reliability"][0];mas={x["ma_window"]:x for x in research["ma_summaries"]};m20=mas.get(20,{});m25=mas.get(25,{})
    idx_rows=[]
    for r in research.get("style_comparison",[]):
        vals=[r[k]["b_mean"] for k in ("ma20","ma25") if k in r and _ok(r[k].get("b_mean"))]
        if vals:idx_rows.append((_avg(vals),r["index_name"]))
    idx_rows.sort();deepest=idx_rows[0] if idx_rows else (None,"—");shallowest=idx_rows[-1] if idx_rows else (None,"—")
    ov=[f"1. <b>三日确认显著过滤噪声，但不是低成本确认。</b> Day1候选最终形成Day3真跌破的比例为 {_pct(sig['day1_success_rate'])}；已经走到Day2后，条件成功率升至 {_pct(sig['day2_success_rate'])}。首日跌破中约 {_pct(sig['fake_rate'])} 最终不会完成三日真跌破。"]
    if m20 and m25:
        ov += [
        f"2. <b>真跌破后的下行仍呈明显厚尾。</b> MA20下甲1中位为 {_pct(m20['a1']['median'])}、P90为 {_pct(m20['a1']['p90_risk'])}；MA25分别为 {_pct(m25['a1']['median'])} 和 {_pct(m25['a1']['p90_risk'])}。常态与极端调整之间存在明显断层。",
        f"3. <b>等待Day3确认会减少剩余下跌，但不会消除尾部风险。</b> MA20甲1/甲3中位从 {_pct(m20['a1']['median'])} 收敛至 {_pct(m20['a3']['median'])}，MA25从 {_pct(m25['a1']['median'])} 收敛至 {_pct(m25['a3']['median'])}。",
        f"4. <b>乙揭示的整轮回撤明显深于单纯从D1往后看的甲。</b> MA20乙中位为 {_pct(m20['b']['median'])}，MA25为 {_pct(m25['b']['median'])}，说明不少事件在Day1以前已经从近10日高点完成了一段回撤。",
        f"5. <b>MA25更慢、事件更少，同时对应的完整回撤更深。</b> 完整事件数MA20为 {m20['n']} 次、MA25为 {m25['n']} 次；乙均值分别为 {_pct(m20['b']['mean'])} 和 {_pct(m25['b']['mean'])}，D1→R3中位分别为 {_num(m20['d1_to_r3']['median'])} 与 {_num(m25['d1_to_r3']['median'])} 个交易日。"]
    if deepest[0] is not None:ov.append(f"6. <b>指数风格差异仍然明显。</b> 按MA20/25的乙平均值综合观察，{deepest[1]}完整回撤最深（约 {_pct(deepest[0])}），{shallowest[1]}相对最浅（约 {_pct(shallowest[0])}）。")
    ps=research.get("period_summary",[])
    if ps:
        worst=min(ps,key=lambda x:x["a1"]["mean"]);best=max(ps,key=lambda x:x["a1"]["mean"])
        ov.append(f"7. <b>市场阶段仍是核心解释变量。</b> {worst['period']}甲1平均跌幅达到 {_pct(worst['a1']['mean'])}，而{best['period']}约为 {_pct(best['a1']['mean'])}；阶段差异明显大于单纯MA20/MA25参数差异。")
    ct=research.get("conduction_top",[])
    if ct:
        top=max(ct,key=lambda x:x["probability"]);ov.append(f"8. <b>跨指数风险表现为共振链条，而不是唯一领先指标。</b> 在样本数不少于20次的组合中，MA{top['ma_window']}下“{top['source_name']}→{top['target_name']}”在{top['horizon']}个交易日窗口内条件触发率约 {_pct(top['probability'])}。")

    depth=[]
    if m20 and m25:
        depth=[
        f"全样本甲1→甲2→甲3的中位剩余风险逐级收敛：MA20为 {_pct(m20['a1']['median'])} → {_pct(m20['a2']['median'])} → {_pct(m20['a3']['median'])}；MA25为 {_pct(m25['a1']['median'])} → {_pct(m25['a2']['median'])} → {_pct(m25['a3']['median'])}。等待确认确实减少后续风险，但不是把尾部风险消除。",
        f"乙的风险尺度明显高于甲1：MA20乙中位 {_pct(m20['b']['median'])}、P90 {_pct(m20['b']['p90_risk'])}；MA25乙中位 {_pct(m25['b']['median'])}、P90 {_pct(m25['b']['p90_risk'])}。乙与甲1/2/3回答的是不同问题，不能混用。",
        f"丙呈现中位接近盈亏平衡、均值更差的特征：MA20中位 {_pct(m20['c']['median'])}、均值 {_pct(m20['c']['mean'])}；MA25中位 {_pct(m25['c']['median'])}、均值 {_pct(m25['c']['mean'])}。少数深跌周期会显著拖累均值。"]

    conditional=[]
    for p in conditional_panels:
        bs=p.get("buckets",[])
        if len(bs)>=2:
            lo,hi=bs[0],bs[-1];conditional.append({"feature":p.get("feature_label",p.get("feature","条件变量")),"text":f"从最低分组到最高分组，甲3中位由 {_pct(lo.get('median_a3'))} 变化至 {_pct(hi.get('median_a3'))}，跌超10%的概率由 {_pct(lo.get('prob_gt_10'))} 变化至 {_pct(hi.get('prob_gt_10'))}。{'该变量呈单调分层，可作为风险等级变量。' if p.get('monotonic') else '该变量存在区分度，但并非严格单调，实盘使用应避免机械阈值。'}"})
    framework=[]
    blocks={b.get('key'):b for b in research.get('conditional_blocks',[])}
    def compare_block(key,label):
        rows=blocks.get(key,{}).get('rows',[])
        for ma in (20,25):
            q=[r for r in rows if r.get('ma_window')==ma and _ok(r.get('a3_median'))]
            if len(q)>=2:
                worst=min(q,key=lambda r:r['a3_median']);best=max(q,key=lambda r:r['a3_median'])
                framework.append(f"{label}（MA{ma}）：风险较高状态“{worst.get('bucket',worst.get('state','—'))}”的甲3中位 {_pct(worst.get('a3_median'))}、跌超10%概率 {_pct(worst.get('prob_a3_10'))}；相对较低状态“{best.get('bucket',best.get('state','—'))}”分别为 {_pct(best.get('a3_median'))} 和 {_pct(best.get('prob_a3_10'))}。")
    compare_block('d3_dev','确认深度');compare_block('d1_d3','三日跌速');compare_block('breadth','市场共振');compare_block('long_trend','长趋势背景')
    esc=blocks.get('escalation',{}).get('rows',[])
    if len(esc)>=2:
        a=min(esc,key=lambda r:r.get('a3_median',0));b=max(esc,key=lambda r:r.get('a3_median',0));framework.append(f"均线升级：{a.get('state')}的甲3中位为 {_pct(a.get('a3_median'))}、乙中位 {_pct(a.get('b_median'))}；{b.get('state')}分别为 {_pct(b.get('a3_median'))} 和 {_pct(b.get('b_median'))}。MA25进一步确认可作为风险升级状态，而不是新的独立事件逻辑。")
    ur=blocks.get('unrecovered',{}).get('rows',[])
    for ma in (20,25):
        r=next((x for x in ur if x.get('ma_window')==ma and x.get('elapsed')==20),None)
        if r:framework.append(f"时间状态（MA{ma}）：D1后20个交易日仍未完成R3的历史事件，剩余修复时间中位约 {_num(r.get('remaining_median'),0)} 日，未来5日内完成R3的比例约 {_pct(r.get('recover_next5_rate'))}。‘迟迟收不回’本身会更新风险判断。")
    framework.append("组合使用顺序：先用Day3偏离与D1→D3跌速判断信号自身强弱，再用多指数广度和MA60/120/200背景判断是否处于系统性弱势；若MA20周期内进一步确认MA25，则视为风险升级；事件发生后再用‘已经多久没收回’动态更新风险。")
    if framework:conditional.append({"feature":"八、从统计报告到风险判断框架","text":" ".join(framework)})

    dur=[]
    if m20 and m25:dur.append(f"均线真收回明显慢于最终见底。MA20 D1→R3中位约 {_num(m20['d1_to_r3']['median'])} 个交易日，MA25约 {_num(m25['d1_to_r3']['median'])} 个交易日；风险释放和趋势修复不是同一个时间点。")
    curves=research.get("recovery_curves",[]);h20=next((x for x in curves if x["days"]==20),None);h60=next((x for x in curves if x["days"]==60),None)
    if h20 and h60:dur.append(f"D1后20个交易日内已有 {_pct(h20['trough_rate'])} 的事件到达最终低点、{_pct(h20['r3_rate'])} 完成R3，同期 {_pct(h20['p1_rate'])} 收复D1价格；到60日时分别为 {_pct(h60['trough_rate'])}、{_pct(h60['r3_rate'])}、{_pct(h60['p1_rate'])}。P1与R3并非同一修复概念。")
    ws=research.get("window_summary",[]);ten=next((x for x in ws if x["days"]==10),None);twenty=next((x for x in ws if x["days"]==20),None)
    if ten and twenty:dur.append(f"固定窗口风险路径显示，D1后10日内最大跌幅中位 {_pct(ten['median_maxdd'])}、P90 {_pct(ten['p90_maxdd'])}；20日窗口扩大到 {_pct(twenty['median_maxdd'])} 和 {_pct(twenty['p90_maxdd'])}。")
    corrs=research.get('duration_correlations',[])
    if corrs:
        c1=_avg([x.get('corr_a1',x.get('corr_r3_a1_risk')) for x in corrs]);cb=_avg([x.get('corr_b',x.get('corr_r3_b_risk')) for x in corrs]);dur.append(f"逐指数×均线相关性综合看，D1→R3与|甲1|相关系数简单平均约 {_num(c1,2)}，与|乙|约 {_num(cb,2)}。这不是因果关系，但支持把‘修复拖延’视为风险状态变量，而不是单纯事后时间标签。")

    ptext=[];detail=research.get("period_detail",[]);pma=research.get('period_ma_summary',[])
    for p in ps:
        rows=[r for r in detail if r["period"]==p["period"]];mrows={r['ma_window']:r for r in pma if r['period']==p['period']};paras=[]
        deep=min(rows,key=lambda x:x["a1"]["mean"]) if rows else None;shallow=max(rows,key=lambda x:x["a1"]["mean"]) if rows else None
        extra=f" 在指数×均线组合中，{deep['index_name']} MA{deep['ma_window']}甲1平均最深（{_pct(deep['a1']['mean'])}），{shallow['index_name']} MA{shallow['ma_window']}相对最浅（{_pct(shallow['a1']['mean'])}）。" if deep and shallow else ""
        paras.append(f"本阶段共 {p['n']} 个完整事件，甲1平均 {_pct(p['a1']['mean'])}、中位 {_pct(p['a1']['median'])}、P90 {_pct(p['a1']['p90_risk'])}；乙平均 {_pct(p['b']['mean'])}，丙中位 {_pct(p['c']['median'])}，D1→R3中位 {_num(p['duration']['median'])} 个交易日。{extra}")
        a=mrows.get(20);b=mrows.get(25)
        if a and b:
            gap=abs(float(a['a1']['mean'])-float(b['a1']['mean']))
            stage=f"该阶段MA20/MA25甲1均值分别为 {_pct(a['a1']['mean'])}/{_pct(b['a1']['mean'])}，差约 {gap*100:.1f} 个百分点；甲3中位分别为 {_pct(a['a3']['median'])}/{_pct(b['a3']['median'])}，D1→R3中位分别为 {_num(a['duration']['median'],0)}/{_num(b['duration']['median'],0)} 日。"
            tail=abs(float(p['a1']['p90_risk']));mean_med=abs(float(p['a1']['mean'])-float(p['a1']['median']))
            if abs(float(p['a1']['mean']))>=.12:stage+=" 系统性风险占主导时，均线长度与风格差异通常会被整体下行压过。"
            elif tail>=.20 and mean_med>=.04:stage+=" 该阶段呈明显分布断裂：多数事件可能较快修复，但少数尾部事件极深；丙中位或短修复时间不能替代路径风险判断，必须同时看P90/P99与极端事件。"
            elif float(p['c']['median'])>0 and float(p['duration']['median'])<=15 and tail<.15:stage+=" 丙中位为正且修复相对快，更接近强趋势中的阶段性回撤，但仍需独立观察尾部。"
            elif float(p['duration']['median'])>=20:stage+=" 平均跌幅未必最深，但修复时间偏长，属于‘拖在均线下方’风险更突出的环境。"
            else:stage+=" 风险深度和修复速度均处于中间区间，风格差异比单一均线参数更值得关注。"
            paras.append(stage)
        for ma in (20,25):
            q=[r for r in rows if r['ma_window']==ma]
            if q:
                vals='；'.join(f"{r['index_name']} {_pct(r['a1']['mean'])}" for r in sorted(q,key=lambda x:x['index_name']))
                paras.append(f"<b>MA{ma}各指数甲1均值：</b>{vals}。")
        ptext.append({"period":p["period"],"paragraphs":paras})

    wide=research.get('conduction_wide',[]);ctext=[]
    for ma in (20,25):
        rows=[x for x in wide if x.get('ma_window')==ma and _ok(x.get('rate_10')) and (x.get('denominator') or 0)>=20]
        if rows:
            top=max(rows,key=lambda x:x['rate_10']);low=min(rows,key=lambda x:x['rate_10']);ctext.append(f"MA{ma}的10日传导中，{top['source_name']}→{top['target_name']}条件触发率较高（{_pct(top['rate_10'])}，N={top['denominator']}，中位滞后 {_num(top.get('median_lag_10'),1)} 日）；样本内较弱组合为{low['source_name']}→{low['target_name']}（{_pct(low['rate_10'])}）。")
            large={'上证指数','上证50','沪深300'};growth={'中证1000','创业板指','科创50'}
            lg=[r['rate_10'] for r in rows if r['source_name'] in large and r['target_name'] in large];gr=[r['rate_10'] for r in rows if r['source_name'] in growth and r['target_name'] in growth]
            if lg or gr:ctext.append(f"MA{ma}内部组平均10日跟随率：大盘组约 {_pct(_avg(lg))}，小盘/成长组约 {_pct(_avg(gr))}。这更像同类风格风险的同步确认，而非稳定单向领先。")
    ctext.append("跨指数矩阵应与市场广度联合使用：单一指数先跌破但其他指数不跟随，和多个指数在短窗口连续确认D3，代表的是不同风险状态。传导率用于共振判断，不作因果预测。")
    return {"overview_conclusions":ov,"depth_commentary":depth,"conditional_commentary":conditional,"conditional_framework":framework,"duration_commentary":dur,"period_commentary":ptext,"conduction_commentary":ctext}

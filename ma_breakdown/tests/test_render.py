from pathlib import Path
from ma_breakdown.src.render import render_report
from ma_breakdown.src.validate import validate_payload


def dummy_payload():
    instruments=[{"key":k,"name":n,"category":"test","sample_start":"2005-01-04","actual_start":"2005-01-04","rows":100} for k,n in [
        ("sse","上证指数"),("sse50","上证50"),("csi300","沪深300"),("csi1000","中证1000"),("chinext","创业板指"),("star50","科创50"),("dividend","中证红利")]]
    research={
      "signal_reliability":[{"scope":"overall","index_name":"全样本","ma_window":None,"day1_candidates":10,"day2_reached":8,"true_breakdowns":6,"day1_success_rate":.6,"day2_success_rate":.75,"fake_rate":.4}],
      "signal_period":[],
      "depth_tables":[{"index_name":"上证指数","ma_window":20,"n":5,"metrics":{"a1":{"n":5,"mean":-.04,"median":-.03,"std":.02,"p25":-.01,"p75":-.05,"p75_risk":-.05,"p90_risk":-.1,"p95_risk":-.12,"p99_risk":-.18,"extreme":-.2,"prob_gt_3":.5,"prob_gt_5":.4,"prob_gt_10":.2,"prob_gt_15":.1,"prob_gt_20":0},"a2":{"n":5,"mean":-.03,"median":-.025,"std":.02,"p90_risk":-.08,"p95_risk":-.1,"p99_risk":-.12,"extreme":-.15,"prob_gt_5":.3,"prob_gt_10":.1},"a3":{"n":5,"mean":-.025,"median":-.02,"std":.02,"p90_risk":-.07,"p95_risk":-.09,"p99_risk":-.11,"extreme":-.13,"prob_gt_5":.2,"prob_gt_10":.1},"b":{"n":5,"mean":-.07,"median":-.06,"std":.03,"p90_risk":-.15,"p95_risk":-.17,"p99_risk":-.19,"extreme":-.2,"prob_gt_10":.3,"prob_gt_20":0}},"c":{"n":5,"mean":-.01,"median":.002,"std":.03,"p10":-.05,"p25":-.02,"p75":.015,"p90":.03,"worst":-.1,"best":.05,"positive_rate":.6,"loss_rate":.4},"d1_to_trough":{"median":6,"p90":15},"d3_to_trough":{"median":4,"p90":12},"d1_to_r3":{"median":18,"p90":35},"d1_to_p1":{"median":12,"p90":80}}],
      "ma_summaries":[{"ma_window":20,"n":5,"a1":{"mean":-.04,"median":-.03,"p90_risk":-.1,"p95_risk":-.12,"p99_risk":-.18},"a2":{"median":-.025},"a3":{"median":-.02},"b":{"mean":-.07,"median":-.06,"p90_risk":-.15},"c":{"mean":-.01,"median":.002,"positive_rate":.6},"d1_to_r3":{"median":18}},{"ma_window":25,"n":4,"a1":{"mean":-.05,"median":-.04,"p90_risk":-.12,"p95_risk":-.14,"p99_risk":-.2},"a2":{"median":-.03},"a3":{"median":-.025},"b":{"mean":-.08,"median":-.07,"p90_risk":-.16},"c":{"mean":-.012,"median":.001,"positive_rate":.5},"d1_to_r3":{"median":20}}],
      "style_comparison":[],"extreme_events":[],
      "duration_rows":[{"index_name":"上证指数","ma_window":20,"n":5,"d1_to_trough":{"mean":8,"median":6,"p75":10,"p90":15,"p95":18,"max":20,"le5":.4,"le10":.7,"le20":1,"gt40":0},"d3_to_trough":{"median":4,"p90":12},"trough_to_r1":{"median":5,"p90":15},"trough_to_r3":{"median":7,"p90":18},"d1_to_r3":{"mean":21,"median":18,"p75":25,"p90":35,"p95":40,"max":45,"le5":.1,"le10":.2,"le20":.6,"gt40":.1},"d1_to_p1":{"median":12,"p90":80},"p1_censored_count":1,"p1_censored_rate":.2}],
      "duration_buckets":{"d1_to_p1":{"label":"D1→价格收复P1","rows":[{"bucket":"≤3","count":1,"rate":.2},{"bucket":"未收复","count":1,"rate":.2}]}},
      "duration_risk_buckets":[{"ma_window":20,"bucket":"≤5","n":2,"a1_mean":-.025,"a1_median":-.02,"a1_p90":-.05,"a3_median":-.01,"b_median":-.04,"c_median":.01,"c_loss_rate":.2}],
      "duration_correlations":[{"index_name":"上证指数","ma_window":20,"n":5,"corr_a1":.5,"corr_a3":.4,"corr_b":.6,"corr_p1_b":.3}],
      "longest_events":[],
      "recovery_curves":[{"days":20,"trough_rate":.8,"r3_rate":.6,"p1_rate":.7}],
      "window_summary":[{"days":10,"n":5,"median_maxdd":-.04,"p90_maxdd":-.1,"median_return":-.01,"p10_return":-.08}],
      "conditional_overall":[{"ma_window":20,"n":5,"prob_a1_5":.4,"prob_a1_10":.2,"prob_a1_15":.1,"prob_a1_20":0,"prob_a3_10":.1,"prob_r3_gt20":.2,"prob_r3_gt40":.1,"c_loss_rate":.4,"worst10_mean":-.15,"worst5_mean":-.2}],
      "conditional_blocks":[
        {"key":"d3_dev","title":"Day3偏离均线深度","rows":[{"ma_window":20,"bucket":"≤-4%","n":2,"a1_median":-.1,"a1_p90":-.15,"a3_median":-.08,"a3_p90":-.12,"prob_a3_10":.5,"b_median":-.14,"c_median":-.03,"c_loss_rate":.6,"d3_to_trough_median":8,"d1_to_r3_median":30}]},
        {"key":"d1_d3","title":"D1→D3跌幅","rows":[]},{"key":"breadth","title":"多指数共振广度","rows":[]},{"key":"long_trend","title":"长期趋势背景","rows":[]},
        {"key":"escalation","title":"MA20→MA25升级","rows":[]},{"key":"unrecovered","title":"已持续未收回","rows":[]}
      ],
      "period_summary":[{"period":"2022至今","label":"test","start":"2022-01-01","end":"2026-07-31","n":5,"a1":{"mean":-.04,"median":-.03,"p90_risk":-.1},"a3":{"mean":-.03},"b":{"mean":-.07,"median":-.06},"c":{"mean":-.01,"median":.002},"duration":{"median":18}}],
      "period_ma_summary":[{"period":"2022至今","label":"test","start":"2022-01-01","end":"2026-07-31","ma_window":20,"n":5,"a1":{"mean":-.04,"median":-.03,"p90_risk":-.1,"p99_risk":-.18},"a2":{"median":-.025},"a3":{"mean":-.03,"median":-.02},"b":{"mean":-.07,"median":-.06},"c":{"median":.002},"duration":{"median":18},"p1":{"median":12}}],
      "period_detail":[],
      "conduction_top":[],
      "conduction_wide":[{"source_name":"上证指数","target_name":"上证50","ma_window":20,"denominator":5,"same_day_rate":.2,"rate_3":.4,"rate_5":.5,"rate_10":.6,"rate_20":.8,"avg_lag_10":3,"median_lag_10":2}],
      "conduction_matrices":{"20":[{"source_name":"上证指数","cells":[{"target_name":"上证50","denominator":5,"rate_3":.4,"rate_5":.5,"rate_10":.6,"rate_20":.8,"median_lag_10":2}]}],"25":[]},
      "ma_validation":[{"index_name":"上证指数","date":"2025-01-10","ma_window":20,"manual":100,"rolling":100,"abs_diff":0,"match":True}],
      "signal_chart":{"periods":[{"name":"2022至今","start":"2022-01-01","end":"2026-07-31"}],"indices":[{"key":"sse","name":"上证指数"}],"prices":{"sse":[["2025-01-01",100,99,98],["2025-01-02",98,99,98.5]]},"events":[{"index_key":"sse","ma_window":20,"d1_date":"2025-01-02","d3_date":"2025-01-06","r1_date":"2025-01-20","r3_date":"2025-01-22","completed":True}]}
    }
    narrative={"overview_conclusions":["1. <b>测试结论。</b> 数据来自当前payload。"],"depth_commentary":["跌幅分析文字。"],"conditional_commentary":["条件风险分析文字。"],"duration_commentary":["持续时间分析文字。"],"period_commentary":[{"period":"2022至今","paragraphs":["时段分析文字。","风格分化文字。"]}],"conduction_commentary":["跨指数分析文字。"]}
    return {
      "meta":{"version":"v0.2","cutoff":"2026-07-31","event_count":1,"ongoing_count":1},
      "instruments":instruments,
      "overview":{"median_a1":-0.04,"p90_a1":-0.12,"extreme_b":-0.30,"median_c":0.002,"day1_success_rate":0.61,"day2_success_rate":0.79},
      "depth_rows":[{"index_name":"上证指数","ma_window":20,"n":5,"metric":"a1","metric_label":"甲1","median":-0.03,"p75_risk":-0.05,"p90_risk":-0.1,"p95_risk":-0.12,"mean":-0.04,"extreme":-0.2,"prob_gt_3":.5,"prob_gt_5":.4,"prob_gt_10":.2,"prob_gt_15":.1,"prob_gt_20":0}],
      "conditional_panels":[{"feature":"d3_ma_dev","feature_label":"D3相对均线偏离","monotonic":True,"buckets":[{"bucket":"Q1","n":10,"median_a3":-.02,"p90_a3":-.05,"prob_gt_5":.1,"prob_gt_10":0,"median_days":3}]}],
      "duration":{"d1_to_trough":{"n":1,"median":8,"p75":14,"p90":25,"mean":10,"max":50},"d1_to_r3":{"n":1,"median":20,"p75":30,"p90":45,"mean":24,"max":80},"d1_to_p1":{"n":1,"median":30,"p75":60,"p90":100,"mean":45,"max":200,"censored_count":1,"censored_rate":.1}},
      "period_rows":[{"year":2025,"ma_window":20,"events":3,"day1_success_rate":.6,"day2_success_rate":.8}],
      "conduction_rows":[{"source_name":"上证指数","target_name":"上证50","ma_window":20,"horizon":10,"denominator":4,"hits":2,"probability":.5}],
      "events":[{"index_name":"上证指数","ma_window":20,"peak_date":"2025-01-01","peak_close":105,"d1_date":"2025-01-10","d1_close":100,"d1_ma":101,"d2_date":"2025-01-13","d2_close":98,"d2_ma":100,"d3_date":"2025-01-14","d3_close":96,"d3_ma":99,"trough_date":"2025-01-20","trough_close":90,"r1_date":"2025-02-01","r1_close":101,"r3_date":"2025-02-03","r3_close":102,"p1_date":"2025-02-10","p1_close":103,"a1":-.1,"a2":-.0816,"a3":-.0625,"b":-.1429,"c":.01,"d1_to_trough":6,"d3_to_trough":4,"trough_to_r1":8,"trough_to_r3":10,"d1_to_r3":18,"d1_to_p1":23,"breadth_ratio":.4,"d3_ma_dev":-.03,"d1_to_d3":-.04}],
      "charts":{"period_svg":"<svg></svg>","duration_svg":"<svg></svg>","window_svg":"<svg></svg>","event_svg":"<svg></svg>"},
      "data_quality":{"status":"PASS","checks":[]},"research":research,"narrative":narrative
    }


def test_validate_payload_requires_correct_index_set_and_counts():
    assert validate_payload(dummy_payload())==[]
    bad=dummy_payload(); bad["instruments"].append({"key":"all","name":"中证全指"})
    assert any("中证全指" in x for x in validate_payload(bad))


def test_render_keeps_eight_pages_v02_definitions_and_v013_design_contract(tmp_path):
    out=tmp_path/"index.html"
    render_report(dummy_payload(),Path("ma_breakdown/templates/report.html"),out)
    html=out.read_text(encoding="utf-8")
    for page in ["overview","design","depth","conditional","duration","period","conduction","events"]:
        assert f'id="page-{page}"' in html
    assert "连续3个交易日收盘低于" in html
    assert "Day1前10个交易日" in html
    assert "甲1" in html and "甲2" in html and "甲3" in html and "乙" in html and "丙" in html
    assert "上证50" in html and "中证全指" not in html
    css=html.split("</style>")[0]
    nav_css=css.split(".nav{")[1].split("}")[0]
    assert "overflow-x:auto" not in nav_css
    assert ".table-scroll" in css and "overflow-x:auto" in css
    assert ".signal-wrap" in css
    assert html.count('class="card"') >= 8
    assert "核心结论" in html and "跌幅分析文字" in html and "持续时间分析文字" in html and "时段分析文字" in html
    assert "{{" not in html and "TODO" not in html


def test_render_restores_full_analysis_and_interactive_signal_curve(tmp_path):
    out=tmp_path/"index.html"
    render_report(dummy_payload(),Path("ma_breakdown/templates/report.html"),out)
    html=out.read_text(encoding="utf-8")
    assert html.count('class="table-scroll"') >= 10
    assert "P99" in html and "甲2" in html and "甲3" in html and "正收益率" in html
    for text in ["计算验证","分布与尾部风险","指数风格差异","三日确认的成本","MA20 vs MA25","极端跌幅事件"]:
        assert text in html
    for text in ["先看概率","Day3偏离均线深度","D1→D3跌幅","多指数共振广度","长期趋势背景","MA20→MA25升级","已持续未收回"]:
        assert text in html
    for text in ["收回速度分布","持续时间与跌幅深度","各指数相关系数","持续时间最长"]:
        assert text in html
    for text in ["阶段总体概览","逐时段解读","阶段 × 指数完整统计","信号曲线"]:
        assert text in html
    for control in ["signalPeriod","signalIndex","signalMa","signalSvg"]:
        assert f'id="{control}"' in html
    assert "真跌破Day1" in html and "真收回R1" in html and "R3确认" in html
    assert "10日跟随率 / 中位滞后" in html and 'class="matrix"' in html
    assert "同日率" in html and "3日率" in html and "5日率" in html and "20日率" in html
    for control in ["eventIndex","eventMa","eventDate","eventReset"]:
        assert f'id="{control}"' in html
    for text in ["Peak价","D1 MA","D2 MA","D3 MA","R1价","R3价","P1价","D3偏离","共振广度"]:
        assert text in html

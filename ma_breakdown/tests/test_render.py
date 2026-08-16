from pathlib import Path
from ma_breakdown.src.render import render_report
from ma_breakdown.src.validate import validate_payload


def dummy_payload():
    instruments=[{"key":k,"name":n,"category":"test","sample_start":"2005-01-04","actual_start":"2005-01-04","rows":100} for k,n in [
        ("sse","上证指数"),("sse50","上证50"),("csi300","沪深300"),("csi1000","中证1000"),("chinext","创业板指"),("star50","科创50"),("dividend","中证红利")]]
    research={
      "signal_reliability":[{"scope":"overall","index_name":"全样本","ma_window":None,"day1_candidates":10,"day2_reached":8,"true_breakdowns":6,"day1_success_rate":.6,"day2_success_rate":.75,"fake_rate":.4}],
      "signal_period":[],
      "depth_tables":[{"index_name":"上证指数","ma_window":20,"n":5,"metrics":{"a1":{"n":5,"mean":-.04,"median":-.03,"std":.02,"p75_risk":-.05,"p90_risk":-.1,"p95_risk":-.12,"p99_risk":-.18,"extreme":-.2,"prob_gt_3":.5,"prob_gt_5":.4,"prob_gt_10":.2,"prob_gt_15":.1,"prob_gt_20":0},"a2":{"n":5,"mean":-.03,"median":-.025,"p90_risk":-.08},"a3":{"n":5,"mean":-.025,"median":-.02,"p90_risk":-.07},"b":{"n":5,"mean":-.07,"median":-.06,"p90_risk":-.15}},"c":{"n":5,"mean":-.01,"median":.002,"std":.03,"p25":-.02,"p75":.015,"worst":-.1,"best":.05,"positive_rate":.6},"d1_to_r3":{"median":18,"p90":35}}],
      "ma_summaries":[{"ma_window":20,"n":5,"a1":{"mean":-.04,"median":-.03,"p90_risk":-.1},"a2":{"median":-.025},"a3":{"median":-.02},"b":{"mean":-.07,"median":-.06,"p90_risk":-.15},"c":{"mean":-.01,"median":.002},"d1_to_r3":{"median":18}},{"ma_window":25,"n":4,"a1":{"mean":-.05,"median":-.04,"p90_risk":-.12},"a2":{"median":-.03},"a3":{"median":-.025},"b":{"mean":-.08,"median":-.07,"p90_risk":-.16},"c":{"mean":-.012,"median":.001},"d1_to_r3":{"median":20}}],
      "style_comparison":[],"extreme_events":[],
      "duration_rows":[{"index_name":"上证指数","ma_window":20,"n":5,"d1_to_trough":{"median":6,"p90":15},"d3_to_trough":{"median":4,"p90":12},"trough_to_r1":{"median":5,"p90":15},"trough_to_r3":{"median":7,"p90":18},"d1_to_r3":{"median":18,"p90":35},"d1_to_p1":{"median":12,"p90":80},"p1_censored_count":1,"p1_censored_rate":.2}],
      "duration_buckets":{"d1_to_p1":{"label":"D1→价格收复P1","rows":[{"bucket":"≤3","count":1,"rate":.2},{"bucket":"未收复","count":1,"rate":.2}]}},
      "recovery_curves":[{"days":20,"trough_rate":.8,"r3_rate":.6,"p1_rate":.7}],
      "window_summary":[{"days":10,"n":5,"median_maxdd":-.04,"p90_maxdd":-.1,"median_return":-.01,"p10_return":-.08}],
      "period_summary":[{"period":"2022至今","label":"test","start":"2022-01-01","end":"2026-07-31","n":5,"a1":{"mean":-.04,"median":-.03,"p90_risk":-.1},"a3":{"mean":-.03},"b":{"mean":-.07,"median":-.06},"c":{"mean":-.01,"median":.002},"duration":{"median":18}}],
      "period_detail":[],"conduction_top":[]
    }
    narrative={"overview_conclusions":["1. <b>测试结论。</b> 数据来自当前payload。"],"depth_commentary":["跌幅分析文字。"],"conditional_commentary":[],"duration_commentary":["持续时间分析文字。"],"period_commentary":[{"period":"2022至今","text":"时段分析文字。"}],"conduction_commentary":["跨指数分析文字。"]}
    return {
      "meta":{"version":"v0.2","cutoff":"2026-07-31","event_count":1,"ongoing_count":1},
      "instruments":instruments,
      "overview":{"median_a1":-0.04,"p90_a1":-0.12,"extreme_b":-0.30,"median_c":0.002,"day1_success_rate":0.61,"day2_success_rate":0.79},
      "depth_rows":[{"index_name":"上证指数","ma_window":20,"n":5,"metric":"a1","metric_label":"甲1","median":-0.03,"p75_risk":-0.05,"p90_risk":-0.1,"p95_risk":-0.12,"mean":-0.04,"extreme":-0.2,"prob_gt_3":.5,"prob_gt_5":.4,"prob_gt_10":.2,"prob_gt_15":.1,"prob_gt_20":0}],
      "conditional_panels":[{"feature":"d3_ma_dev","feature_label":"D3相对均线偏离","monotonic":True,"buckets":[{"bucket":"Q1","n":10,"median_a3":-.02,"p90_a3":-.05,"prob_gt_5":.1,"prob_gt_10":0,"median_days":3}]}],
      "duration":{"d1_to_trough":{"n":1,"median":8,"p75":14,"p90":25,"mean":10,"max":50},"d1_to_r3":{"n":1,"median":20,"p75":30,"p90":45,"mean":24,"max":80},"d1_to_p1":{"n":1,"median":30,"p75":60,"p90":100,"mean":45,"max":200,"censored_count":1,"censored_rate":.1}},
      "period_rows":[{"year":2025,"ma_window":20,"events":3,"day1_success_rate":.6,"day2_success_rate":.8}],
      "conduction_rows":[{"source_name":"上证指数","target_name":"上证50","ma_window":20,"horizon":10,"denominator":4,"hits":2,"probability":.5}],
      "events":[{"index_name":"上证指数","ma_window":20,"peak_date":"2025-01-01","d1_date":"2025-01-10","d2_date":"2025-01-13","d3_date":"2025-01-14","trough_date":"2025-01-20","r1_date":"2025-02-01","r3_date":"2025-02-03","p1_date":"2025-02-10","a1":-.05,"a2":-.04,"a3":-.03,"b":-.08,"c":.01,"d1_to_trough":6,"d1_to_r3":18,"d1_to_p1":23,"breadth_ratio":.4}],
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

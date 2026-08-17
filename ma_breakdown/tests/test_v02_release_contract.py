import pandas as pd

from ma_breakdown.src.metrics import enrich_event_metrics
from ma_breakdown.src.release_patch import apply_v02_release_patch


def _prices():
    closes = [105.0, 110.0, 99.0, 96.0, 94.0, 90.0, 95.0, 101.0, 108.0, 111.0]
    return pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=len(closes)), "close": closes})


def _event():
    p = _prices()
    return pd.DataFrame([{
        "d1_idx": 2, "d2_idx": 3, "d3_idx": 4,
        "d1_date": p.at[2, "date"], "d2_date": p.at[3, "date"], "d3_date": p.at[4, "date"],
        "d1_close": 99.0, "d2_close": 96.0, "d3_close": 94.0,
        "d1_ma": 100.0, "d2_ma": 100.0, "d3_ma": 100.0,
        "peak_idx": 1, "peak_date": p.at[1, "date"], "peak_close": 110.0,
        "trough_idx": 5, "trough_date": p.at[5, "date"], "trough_close": 90.0,
        "r1_idx": 6, "r2_idx": 7, "r3_idx": 8,
        "r1_date": p.at[6, "date"], "r2_date": p.at[7, "date"], "r3_date": p.at[8, "date"],
        "r1_close": 95.0, "r2_close": 101.0, "r3_close": 108.0,
    }])


def _payload():
    base=enrich_event_metrics(_event(),_prices())
    a=base.copy();a["ma_window"]=20;a["index_name"]="测试指数A"
    b=base.copy();b["ma_window"]=25;b["index_name"]="测试指数A"
    events=pd.concat([a,b],ignore_index=True).to_dict("records")
    return {
        "meta":{"version":"v0.2","cutoff":"2026-07-31","event_count":2,"ongoing_count":0},
        "overview":{"day1_success_rate":.639,"median_a1":-.041,"p90_a1":-.165},
        "events":events,
        "research":{"duration_correlations":[]},
        "charts":{"duration_svg":"","window_svg":""},
    }


def _shell():
    return '''<!doctype html><style></style><body><nav class="nav" id="mainNav"><a href="#overview" data-page="overview">研究概览</a><a href="#design" data-page="design">研究设计</a><a href="#depth" data-page="depth">跌幅深度分析</a><a href="#conditional" data-page="conditional">条件风险分析</a><a href="#duration" data-page="duration">持续时间分析</a><a href="#period" data-page="period">时段与信号图表</a><a href="#conduction" data-page="conduction">跨指数传导</a><a href="#events" data-page="events">事件明细</a></nav><main><div class="page" id="page-overview">old overview</div><div class="page" id="page-design"><b>P1：</b>Day1以后第一次Close≥D1 Close，不以R3作为搜索终点。</div><div class="page" id="page-depth">depth</div><div class="page" id="page-conditional">conditional</div><div class="page" id="page-duration">old duration</div><div class="page" id="page-period">period</div><div class="page" id="page-conduction">conduction</div><div class="page" id="page-events">events</div></main></body>'''


def test_p1_uses_d0_price_anchor_not_d1_price():
    enriched = enrich_event_metrics(_event(), _prices())
    row = enriched.iloc[0]
    assert row["d0_idx"] == 1
    assert row["d0_close"] == 110.0
    assert row["p1_idx"] == 9
    assert row["p1_close"] == 111.0
    assert row["d1_to_p1"] == 7


def test_release_output_contains_approved_overview_and_duration_contract():
    html=apply_v02_release_patch(_shell(),_payload())
    assert "持续时间分析：时间不是一个数字，而是三种不同的风险时钟" in html
    assert "D1前一交易日D0" in html
    assert "Close≥D1 Close" not in html
    assert "趋势修复和价格修复必须分开" in html
    assert "20日仍未R3" in html and "40日仍未R3" in html
    assert html.index('data-page="duration"') < html.index('data-page="conditional"')


def test_release_output_keeps_all_eight_pages():
    html=apply_v02_release_patch(_shell(),_payload())
    for marker in [
        'id="page-overview"','id="page-design"','id="page-depth"','id="page-duration"',
        'id="page-conditional"','id="page-period"','id="page-conduction"','id="page-events"',
        "V02_RELEASE_CONTRACT",
    ]:
        assert marker in html, marker

from pathlib import Path

import pandas as pd

from ma_breakdown.src.metrics import enrich_event_metrics


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


def test_p1_uses_d0_price_anchor_not_d1_price():
    enriched = enrich_event_metrics(_event(), _prices())
    row = enriched.iloc[0]
    assert row["d0_idx"] == 1
    assert row["d0_close"] == 110.0
    assert row["p1_idx"] == 9
    assert row["p1_close"] == 111.0
    assert row["d1_to_p1"] == 7


def test_release_template_contains_approved_overview_and_duration_contract():
    template = Path("ma_breakdown/templates/report.html").read_text(encoding="utf-8")
    assert "持续时间分析：时间不是一个数字，而是三种不同的风险时钟" in template
    assert "D1前一交易日D0" in template
    assert "Close≥D1 Close" not in template
    assert "趋势修复和价格修复必须分开" in template
    assert "20日仍未R3" in template and "40日仍未R3" in template
    assert template.index('data-page="duration"') < template.index('data-page="conditional"')


def test_release_template_keeps_full_report_interactions():
    template = Path("ma_breakdown/templates/report.html").read_text(encoding="utf-8")
    for marker in [
        'id="page-overview"', 'id="page-design"', 'id="page-depth"', 'id="page-duration"',
        'id="page-conditional"', 'id="page-period"', 'id="page-conduction"', 'id="page-events"',
        "report-details", "阶段风险地图", "严格延迟", "event-detail", "sortable"
    ]:
        assert marker in template, marker

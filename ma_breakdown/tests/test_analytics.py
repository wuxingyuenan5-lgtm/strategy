import pandas as pd
import numpy as np
from ma_breakdown.src.analytics import add_risk_features, select_conditional_panels, signal_period_stats, conduction_matrix

def make_prices():
    n=260
    dates=pd.bdate_range("2025-01-01",periods=n)
    close=pd.Series(np.linspace(120,90,n)+np.sin(np.arange(n)/8)*2)
    df=pd.DataFrame({"date":dates,"close":close})
    for w in (20,25,60,120,200): df[f"ma{w}"]=df.close.rolling(w).mean()
    return df

def test_add_risk_features_has_required_breakdown_state_fields():
    p=make_prices(); i=220
    ev=pd.DataFrame([{"d1_idx":i,"d3_idx":i+2,"d1_close":float(p.at[i,'close']),"d3_close":float(p.at[i+2,'close']),"d3_ma":float(p.at[i+2,'ma20']),"peak_close":float(p.loc[i-10:i-1,'close'].max()),"a3":-0.08,"d3_to_trough":5}])
    out=add_risk_features(ev,p,20)
    for c in ["d3_ma_dev","d1_to_d3","peak_to_d1","peak_to_d3","ma_slope_5","below_ma60","below_ma120","below_ma200","ret20","ret60","rv20","rv20_pct"]:
        assert c in out.columns

def test_conditional_selector_returns_bucket_risk_stats():
    n=120
    df=pd.DataFrame({"a3":np.linspace(-.01,-.20,n),"d3_to_trough":np.linspace(2,20,n),"d3_ma_dev":np.linspace(-.005,-.06,n),"rv20_pct":np.linspace(.1,.9,n),"breadth_ratio":np.linspace(.1,1,n)})
    panels=select_conditional_panels(df,["d3_ma_dev","rv20_pct","breadth_ratio"],max_panels=2,min_bucket=10)
    assert len(panels)==2
    assert all("buckets" in p and len(p["buckets"])>=3 for p in panels)
    assert {"median_a3","p90_a3","prob_gt_10","median_days"}.issubset(panels[0]["buckets"][0])

def test_signal_period_stats_reports_day1_day2_success():
    rows=[]
    for year,true in [(2024,1),(2024,0),(2025,1)]: rows.append({"year":year,"day1_candidates":10,"day2_reached":8,"true_breakdowns":6 if true else 4})
    out=signal_period_stats(pd.DataFrame(rows))
    assert "day1_success_rate" in out.columns and "day2_success_rate" in out.columns

def test_conduction_uses_common_observable_denominator():
    dates=pd.bdate_range("2026-01-01",periods=30)
    price_dates={"A":dates,"B":dates[5:]}
    events=pd.DataFrame([
        {"index_key":"A","ma_window":20,"d3_date":dates[2]},
        {"index_key":"A","ma_window":20,"d3_date":dates[8]},
        {"index_key":"A","ma_window":20,"d3_date":dates[15]},
        {"index_key":"B","ma_window":20,"d3_date":dates[10]},
    ])
    out=conduction_matrix(events,price_dates,horizon=5,ma_window=20)
    row=out[(out.source=="A")&(out.target=="B")].iloc[0]
    assert row.denominator==2
    assert row.hits==1
    assert row.probability==0.5

def test_breadth_uses_only_indices_observable_on_event_date():
    from ma_breakdown.src.analytics import add_breadth
    dates={"A":pd.bdate_range("2005-01-03",periods=100),"B":pd.bdate_range("2005-01-03",periods=100),"C":pd.bdate_range("2010-01-04",periods=100)}
    d=pd.Timestamp("2005-02-01")
    ev=pd.DataFrame([{"index_key":"A","ma_window":20,"d3_date":d,"r3_date":d+pd.offsets.BDay(5)}])
    out=add_breadth(ev,availability=dates)
    assert out.iloc[0].breadth_count==1
    assert out.iloc[0].breadth_denominator==2
    assert out.iloc[0].breadth_ratio==0.5

import pandas as pd
import numpy as np
from ma_breakdown.src.metrics import enrich_event_metrics, depth_summary, duration_summary, window_path_stats

def prices():
    closes=[110,108,106,104,102,101,100,105,103,107, 99,96,94,90,91,95,97,98,99,100,101,102,103,106,108]
    return pd.DataFrame({"date":pd.bdate_range("2026-01-01",periods=len(closes)),"close":closes})

def event():
    p=prices()
    return pd.DataFrame([{
        "d1_idx":10,"d2_idx":11,"d3_idx":12,"d1_date":p.at[10,"date"],"d2_date":p.at[11,"date"],"d3_date":p.at[12,"date"],
        "d1_close":99.0,"d2_close":96.0,"d3_close":94.0,"d1_ma":100.0,"d2_ma":100.0,"d3_ma":100.0,
        "peak_idx":0,"peak_date":p.at[0,"date"],"peak_close":110.0,
        "trough_idx":13,"trough_date":p.at[13,"date"],"trough_close":90.0,
        "r1_idx":17,"r2_idx":18,"r3_idx":19,"r1_date":p.at[17,"date"],"r2_date":p.at[18,"date"],"r3_date":p.at[19,"date"],
        "r1_close":98.0,"r2_close":99.0,"r3_close":100.0,
    }])

def test_enrich_calculates_a_b_c_and_p1_beyond_r3():
    e=enrich_event_metrics(event(),prices())
    r=e.iloc[0]
    assert np.isclose(r["a1"],90/99-1)
    assert np.isclose(r["a2"],90/96-1)
    assert np.isclose(r["a3"],90/94-1)
    assert np.isclose(r["b"],90/110-1)
    assert np.isclose(r["c"],98/99-1)
    assert r["d0_idx"]==9 and np.isclose(r["d0_close"],107.0)
    assert r["p1_idx"]==24
    assert r["d1_to_p1"]==14
    assert r["d1_to_trough"]==3 and r["d3_to_trough"]==1
    assert r["trough_to_r1"]==4 and r["trough_to_r3"]==6 and r["d1_to_r3"]==9

def test_p1_right_censor_is_preserved():
    p=prices().iloc[:18].copy()
    ev=event().copy(); ev["r1_idx"]=14;ev["r2_idx"]=15;ev["r3_idx"]=16
    ev["r1_date"]=p.at[14,"date"];ev["r2_date"]=p.at[15,"date"];ev["r3_date"]=p.at[16,"date"]
    ev["r1_close"]=91.;ev["r2_close"]=95.;ev["r3_close"]=97.
    e=enrich_event_metrics(ev,p)
    assert bool(e.iloc[0]["p1_censored"])
    assert pd.isna(e.iloc[0]["d1_to_p1"])

def test_depth_summary_uses_tail_risk_quantiles_and_probabilities():
    df=pd.DataFrame({"a1":[-0.01,-0.03,-0.05,-0.10,-0.20]})
    s=depth_summary(df,["a1"]).iloc[0]
    assert s["metric"]=="a1"
    assert s["median"]==-0.05
    assert s["p90_risk"] <= s["median"]
    assert np.isclose(s["prob_gt_10"],0.4)

def test_duration_summary_contains_requested_fields():
    e=enrich_event_metrics(event(),prices())
    d=duration_summary(e)
    assert "d1_to_trough" in d and "d1_to_p1" in d
    assert d["d1_to_trough"]["median"]==3
    assert "p95" in d["d1_to_p1"]

def test_window_path_stats_computes_return_and_max_drawdown():
    e=enrich_event_metrics(event(),prices())
    w=window_path_stats(e,prices(),horizons=(3,5))
    r=w.iloc[0]
    assert np.isclose(r["ret_3"],90/99-1)
    assert np.isclose(r["maxdd_3"],90/99-1)

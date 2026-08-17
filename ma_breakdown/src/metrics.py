from __future__ import annotations
import numpy as np
import pandas as pd

DEPTH_METRICS=("a1","a2","a3","b")
DURATION_FIELDS=("d1_to_trough","d3_to_trough","trough_to_r1","trough_to_r3","d1_to_r3","d1_to_p1")

def enrich_event_metrics(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    x=events.copy()
    x["a1"]=x["trough_close"]/x["d1_close"]-1
    x["a2"]=x["trough_close"]/x["d2_close"]-1
    x["a3"]=x["trough_close"]/x["d3_close"]-1
    x["b"]=x["trough_close"]/x["peak_close"]-1
    x["c"]=x["r1_close"]/x["d1_close"]-1
    x["d1_to_trough"]=(x["trough_idx"]-x["d1_idx"]).astype(int)
    x["d3_to_trough"]=(x["trough_idx"]-x["d3_idx"]).clip(lower=0).astype(int)
    x["trough_before_or_on_d3"]=x["trough_idx"]<=x["d3_idx"]
    x["trough_to_r1"]=(x["r1_idx"]-x["trough_idx"]).clip(lower=0).astype(int)
    x["trough_to_r3"]=(x["r3_idx"]-x["trough_idx"]).clip(lower=0).astype(int)
    x["d1_to_r3"]=(x["r3_idx"]-x["d1_idx"]).astype(int)
    p=prices.sort_values("date").reset_index(drop=True)
    d0_idx=[];d0_date=[];d0_close=[]
    p1_idx=[];p1_date=[];p1_close=[];censored=[];d1_to_p1=[]
    for _,r in x.iterrows():
        d1i=int(r["d1_idx"]); d0i=d1i-1
        if d0i < 0:
            raise ValueError("D1 event has no prior trading day for D0 anchor")
        anchor=float(p.at[d0i,"close"])
        d0_idx.append(d0i);d0_date.append(p.at[d0i,"date"]);d0_close.append(anchor)
        start=d1i+1
        candidates=p.iloc[start:]
        hit=candidates[candidates["close"]>=anchor]
        if hit.empty:
            p1_idx.append(np.nan);p1_date.append(pd.NaT);p1_close.append(np.nan);censored.append(True);d1_to_p1.append(np.nan)
        else:
            idx=int(hit.index[0]);p1_idx.append(idx);p1_date.append(p.at[idx,"date"]);p1_close.append(float(p.at[idx,"close"]));censored.append(False);d1_to_p1.append(idx-d1i)
    x["d0_idx"]=d0_idx;x["d0_date"]=d0_date;x["d0_close"]=d0_close
    x["p1_idx"]=p1_idx;x["p1_date"]=p1_date;x["p1_close"]=p1_close;x["p1_censored"]=censored;x["d1_to_p1"]=d1_to_p1
    return x

def _risk_quantile(s: pd.Series, severity: float) -> float:
    return float(s.quantile(1-severity))

def depth_summary(events: pd.DataFrame, metrics=DEPTH_METRICS) -> pd.DataFrame:
    rows=[]
    for m in metrics:
        if m not in events: continue
        s=pd.to_numeric(events[m],errors="coerce").dropna()
        if s.empty: continue
        rows.append({
            "metric":m,"n":int(s.size),"median":float(s.median()),"mean":float(s.mean()),
            "p75_risk":_risk_quantile(s,.75),"p90_risk":_risk_quantile(s,.90),"p95_risk":_risk_quantile(s,.95),
            "extreme":float(s.min()),
            "prob_gt_3":float((s<=-.03).mean()),"prob_gt_5":float((s<=-.05).mean()),
            "prob_gt_10":float((s<=-.10).mean()),"prob_gt_15":float((s<=-.15).mean()),"prob_gt_20":float((s<=-.20).mean()),
        })
    return pd.DataFrame(rows)

def duration_summary(events: pd.DataFrame) -> dict:
    out={}
    for col in DURATION_FIELDS:
        if col not in events: continue
        s=pd.to_numeric(events[col],errors="coerce").dropna()
        if s.empty:
            out[col]={"n":0}; continue
        out[col]={"n":int(s.size),"median":float(s.median()),"p75":float(s.quantile(.75)),"p90":float(s.quantile(.90)),"p95":float(s.quantile(.95)),"mean":float(s.mean()),"max":float(s.max())}
        if col=="d1_to_p1":
            out[col]["censored_count"]=int(events["p1_censored"].sum()) if "p1_censored" in events else 0
            out[col]["censored_rate"]=float(events["p1_censored"].mean()) if "p1_censored" in events and len(events) else 0.0
    return out

def duration_buckets(events: pd.DataFrame, col: str) -> list[dict]:
    s=pd.to_numeric(events[col],errors="coerce")
    bins=[-np.inf,3,5,10,20,40,60,np.inf]; labels=["≤3","4–5","6–10","11–20","21–40","41–60",">60"]
    cats=pd.cut(s,bins=bins,labels=labels); counts=cats.value_counts(sort=False); denom=s.notna().sum()
    rows=[{"bucket":str(k),"count":int(v),"rate":float(v/denom) if denom else 0.0} for k,v in counts.items()]
    if col=="d1_to_p1" and "p1_censored" in events:
        n=int(events["p1_censored"].sum()); rows.append({"bucket":"未收复","count":n,"rate":float(n/len(events)) if len(events) else 0.0})
    return rows

def cumulative_completion(events: pd.DataFrame, col: str, horizons=(3,5,10,20,40,60,120)) -> list[dict]:
    s=pd.to_numeric(events[col],errors="coerce"); denom=len(events)
    return [{"days":int(h),"rate":float((s<=h).sum()/denom) if denom else 0.0} for h in horizons]

def window_path_stats(events: pd.DataFrame, prices: pd.DataFrame, horizons=(3,5,10,20,40,60)) -> pd.DataFrame:
    p=prices.sort_values("date").reset_index(drop=True); rows=[]
    for ei,e in events.iterrows():
        r={"event_row":ei}; start=int(e["d1_idx"]); base=float(e["d1_close"])
        for h in horizons:
            end=start+int(h)
            if end>=len(p):
                r[f"ret_{h}"]=np.nan;r[f"maxdd_{h}"]=np.nan;continue
            r[f"ret_{h}"]=float(p.at[end,"close"]/base-1)
            r[f"maxdd_{h}"]=float(p.loc[start:end,"close"].min()/base-1)
        rows.append(r)
    return pd.DataFrame(rows)

from __future__ import annotations
import math
import numpy as np
import pandas as pd

def _rolling_pct_rank(s: pd.Series, window: int=252) -> pd.Series:
    def rank_last(a):
        a=np.asarray(a,dtype=float); v=a[-1]; a=a[~np.isnan(a)]
        if len(a)==0 or np.isnan(v): return np.nan
        return float((a<=v).mean())
    return s.rolling(window, min_periods=min(60,window)).apply(rank_last, raw=True)

def prepare_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    p=prices.sort_values("date").reset_index(drop=True).copy()
    for w in (20,25,60,120,200):
        c=f"ma{w}"
        if c not in p: p[c]=p["close"].rolling(w,min_periods=w).mean()
    ret=p["close"].pct_change()
    p["ret20"]=p["close"].pct_change(20); p["ret60"]=p["close"].pct_change(60)
    p["rv20"]=ret.rolling(20,min_periods=20).std()*math.sqrt(252)
    p["rv20_pct"]=_rolling_pct_rank(p["rv20"],252)
    return p

def add_risk_features(events: pd.DataFrame, prices: pd.DataFrame, ma_window: int) -> pd.DataFrame:
    if events.empty: return events.copy()
    p=prepare_price_features(prices); out=events.copy()
    out["d3_ma_dev"]=out["d3_close"]/out["d3_ma"]-1
    out["d1_to_d3"]=out["d3_close"]/out["d1_close"]-1
    out["peak_to_d1"]=out["d1_close"]/out["peak_close"]-1
    out["peak_to_d3"]=out["d3_close"]/out["peak_close"]-1
    vals={k:[] for k in ["ma_slope_5","below_ma60","below_ma120","below_ma200","ret20","ret60","rv20","rv20_pct"]}
    ma_col=f"ma{ma_window}"
    for _,r in out.iterrows():
        i=int(r["d3_idx"]); cur_ma=p.at[i,ma_col] if i<len(p) else np.nan; old_ma=p.at[i-5,ma_col] if i>=5 else np.nan
        vals["ma_slope_5"].append(float(cur_ma/old_ma-1) if pd.notna(cur_ma) and pd.notna(old_ma) and old_ma else np.nan)
        for w in (60,120,200):
            m=p.at[i,f"ma{w}"] if i<len(p) else np.nan
            vals[f"below_ma{w}"].append(bool(p.at[i,"close"]<m) if pd.notna(m) else np.nan)
        for c in ("ret20","ret60","rv20","rv20_pct"):
            vals[c].append(float(p.at[i,c]) if i<len(p) and pd.notna(p.at[i,c]) else np.nan)
    for k,v in vals.items(): out[k]=v
    return out

def _bucket_stats(df: pd.DataFrame, feature: str, min_bucket: int):
    x=df[[feature,"a3","d3_to_trough"]].dropna().copy()
    if len(x)<min_bucket*3 or x[feature].nunique()<3: return None
    try: x["bucket"]=pd.qcut(x[feature],q=4,duplicates="drop")
    except ValueError: return None
    rows=[]
    for bucket,g in x.groupby("bucket",observed=True,sort=True):
        if len(g)<min_bucket: continue
        a=g["a3"]
        rows.append({"bucket":str(bucket),"n":int(len(g)),"feature_median":float(g[feature].median()),
                     "median_a3":float(a.median()),"p90_a3":float(a.quantile(.10)),
                     "prob_gt_5":float((a<=-.05).mean()),"prob_gt_10":float((a<=-.10).mean()),
                     "median_days":float(g["d3_to_trough"].median())})
    if len(rows)<3: return None
    meds=np.array([r["median_a3"] for r in rows]); dif=np.diff(meds)
    monotonic=bool(np.all(dif>=-1e-12) or np.all(dif<=1e-12)); separation=float(meds.max()-meds.min())
    prob_sep=float(max(r["prob_gt_10"] for r in rows)-min(r["prob_gt_10"] for r in rows))
    return {"feature":feature,"score":separation+0.10*prob_sep+(0.01 if monotonic else 0),"monotonic":monotonic,"buckets":rows}

def select_conditional_panels(events: pd.DataFrame, features: list[str], max_panels: int=4, min_bucket: int=15) -> list[dict]:
    panels=[]
    for f in features:
        if f not in events: continue
        p=_bucket_stats(events,f,min_bucket)
        if p: panels.append(p)
    panels.sort(key=lambda z:(z["monotonic"],z["score"]),reverse=True)
    return panels[:max_panels]

def signal_period_stats(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    if candidate_rows.empty: return candidate_rows.copy()
    g=candidate_rows.groupby("year",as_index=False)[["day1_candidates","day2_reached","true_breakdowns"]].sum()
    g["day1_success_rate"]=g["true_breakdowns"]/g["day1_candidates"].replace(0,np.nan)
    g["day2_success_rate"]=g["true_breakdowns"]/g["day2_reached"].replace(0,np.nan)
    return g

def conduction_matrix(events: pd.DataFrame, price_dates: dict[str,pd.DatetimeIndex|pd.Series], horizon: int=10, ma_window: int=20) -> pd.DataFrame:
    ev=events[events["ma_window"]==ma_window].copy(); keys=sorted(price_dates); rows=[]
    for source in keys:
        src=ev[ev["index_key"]==source]
        for target in keys:
            if source==target: continue
            td=pd.DatetimeIndex(pd.to_datetime(price_dates[target])).sort_values().unique()
            target_events=pd.DatetimeIndex(pd.to_datetime(ev.loc[ev["index_key"]==target,"d3_date"]))
            denom=hits=0
            for d in pd.to_datetime(src["d3_date"]):
                pos=int(td.searchsorted(d,side="left"))
                if pos>=len(td) or td[pos] < d: continue
                endpos=pos+horizon
                if endpos>=len(td) or d<td[0]: continue
                denom += 1; end=td[endpos]
                if ((target_events>=d)&(target_events<=end)).any(): hits += 1
            rows.append({"source":source,"target":target,"ma_window":ma_window,"horizon":horizon,
                         "denominator":denom,"hits":hits,"probability":float(hits/denom) if denom else np.nan})
    return pd.DataFrame(rows)

def add_breadth(events: pd.DataFrame, availability: dict[str, pd.DatetimeIndex|pd.Series]) -> pd.DataFrame:
    out=events.copy(); counts=[]; denoms=[]; ratios=[]; ranges={}
    for key,dates in availability.items():
        di=pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
        if len(di): ranges[key]=(di[0],di[-1])
    d3=pd.to_datetime(out["d3_date"]) if not out.empty else pd.Series(dtype="datetime64[ns]")
    r3=pd.to_datetime(out["r3_date"]) if not out.empty else pd.Series(dtype="datetime64[ns]")
    for _,r in out.iterrows():
        d=pd.Timestamp(r["d3_date"]); w=r["ma_window"]
        observable={k for k,(lo,hi) in ranges.items() if lo <= d <= hi}
        active=out[(out["ma_window"]==w)&(d3<=d)&(r3>=d)&(out["index_key"].isin(observable))]
        c=int(active["index_key"].nunique()); denom=len(observable)
        counts.append(c); denoms.append(denom); ratios.append(c/denom if denom else np.nan)
    out["breadth_count"]=counts;out["breadth_denominator"]=denoms;out["breadth_ratio"]=ratios
    return out

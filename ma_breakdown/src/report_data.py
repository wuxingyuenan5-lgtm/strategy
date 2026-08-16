from __future__ import annotations
import numpy as np
import pandas as pd

PERIODS=[
("2005-2007早期样本","早期样本","2005-01-04","2007-12-31"),
("2008危机期","全球金融危机","2008-01-01","2008-12-31"),
("2009反弹期","四万亿反弹","2009-01-01","2010-12-31"),
("震荡下行期","漫长震荡下行","2011-01-01","2014-06-30"),
("2015牛熊","杠杆牛到股灾","2014-07-01","2016-02-29"),
("蓝筹白马期","核心资产行情","2016-03-01","2018-12-31"),
("2019-2021结构牛","结构性牛市","2019-01-01","2021-12-31"),
("2022至今","价值回归+小微盘","2022-01-01","2026-07-31"),]
DEPTH=("a1","a2","a3","b")
DURATIONS=(("d1_to_trough","D1→最终低点"),("d3_to_trough","D3→最终低点"),("trough_to_r1","Trough→R1"),("trough_to_r3","Trough→R3"),("d1_to_r3","D1→真收回R3"),("d1_to_p1","D1→价格收复P1"))

def _f(v): return None if v is None or pd.isna(v) else float(v)
def _stats(s):
    s=pd.to_numeric(pd.Series(s),errors="coerce").dropna()
    if s.empty:return {"n":0}
    return {"n":int(len(s)),"mean":_f(s.mean()),"median":_f(s.median()),"std":_f(s.std(ddof=1)),"p75_risk":_f(s.quantile(.25)),"p90_risk":_f(s.quantile(.10)),"p95_risk":_f(s.quantile(.05)),"p99_risk":_f(s.quantile(.01)),"extreme":_f(s.min()),"prob_gt_3":_f((s<=-.03).mean()),"prob_gt_5":_f((s<=-.05).mean()),"prob_gt_10":_f((s<=-.10).mean()),"prob_gt_15":_f((s<=-.15).mean()),"prob_gt_20":_f((s<=-.20).mean())}
def _cstats(s):
    s=pd.to_numeric(pd.Series(s),errors="coerce").dropna()
    if s.empty:return {"n":0}
    return {"n":int(len(s)),"mean":_f(s.mean()),"median":_f(s.median()),"std":_f(s.std(ddof=1)),"p25":_f(s.quantile(.25)),"p75":_f(s.quantile(.75)),"worst":_f(s.min()),"best":_f(s.max()),"positive_rate":_f((s>0).mean())}
def _duration_stats(s):
    s=pd.to_numeric(pd.Series(s),errors="coerce").dropna()
    if s.empty:return {"n":0}
    return {"n":int(len(s)),"mean":_f(s.mean()),"median":_f(s.median()),"p75":_f(s.quantile(.75)),"p90":_f(s.quantile(.90)),"p95":_f(s.quantile(.95)),"max":_f(s.max()),"le3":_f((s<=3).mean()),"le5":_f((s<=5).mean()),"le10":_f((s<=10).mean()),"le20":_f((s<=20).mean()),"gt40":_f((s>40).mean()),"gt60":_f((s>60).mean())}
def _assign_period(d):
    d=pd.Timestamp(d)
    for name,label,start,end in PERIODS:
        if pd.Timestamp(start)<=d<=pd.Timestamp(end):return name
    return None

def build_research_payload(events,candidates,window_rows,conduction,instruments,cutoff):
    ev=events.copy();ca=candidates.copy();win=window_rows.copy() if window_rows is not None else pd.DataFrame();con=conduction.copy() if conduction is not None else pd.DataFrame()
    for c in ["peak_date","d1_date","d3_date","trough_date","r1_date","r3_date","p1_date"]:
        if c in ev:ev[c]=pd.to_datetime(ev[c],errors="coerce")
    if "d1_date" in ca:ca["d1_date"]=pd.to_datetime(ca["d1_date"],errors="coerce")
    name_map={x["key"]:x["name"] for x in instruments}
    sig=[]
    def add_sig(scope,g,index_name="全样本",ma=None):
        d1=len(g);d2=int(g["reached_day2"].sum()) if d1 else 0;true=int(g["true_breakdown"].sum()) if d1 else 0
        sig.append({"scope":scope,"index_name":index_name,"ma_window":ma,"day1_candidates":d1,"day2_reached":d2,"true_breakdowns":true,"day1_success_rate":_f(true/d1) if d1 else None,"day2_success_rate":_f(true/d2) if d2 else None,"fake_rate":_f(1-true/d1) if d1 else None})
    add_sig("overall",ca)
    for (idx,ma),g in ca.groupby(["index_key","ma_window"]):add_sig("index_ma",g,name_map.get(idx,idx),int(ma))
    depth=[];ma_summ=[]
    for (idx,ma),g in ev.groupby(["index_key","ma_window"]):
        depth.append({"index_key":idx,"index_name":name_map.get(idx,idx),"ma_window":int(ma),"n":int(len(g)),"metrics":{m:_stats(g[m]) for m in DEPTH},"c":_cstats(g["c"]),"d1_to_r3":_duration_stats(g["d1_to_r3"])})
    for ma,g in ev.groupby("ma_window"):
        ma_summ.append({"ma_window":int(ma),"n":int(len(g)),"a1":_stats(g.a1),"a2":_stats(g.a2),"a3":_stats(g.a3),"b":_stats(g.b),"c":_cstats(g.c),"d1_to_r3":_duration_stats(g.d1_to_r3)})
    style=[]
    for idx,g in ev.groupby("index_key"):
        rec={"index_key":idx,"index_name":name_map.get(idx,idx)}
        for ma in (20,25):
            q=g[g.ma_window==ma]
            if len(q):rec[f"ma{ma}"]={"n":int(len(q)),"a1_mean":_f(q.a1.mean()),"a1_median":_f(q.a1.median()),"a3_mean":_f(q.a3.mean()),"b_mean":_f(q.b.mean()),"c_mean":_f(q.c.mean()),"c_median":_f(q.c.median()),"duration_median":_f(q.d1_to_r3.median())}
        style.append(rec)
    cols=["index_name","ma_window","peak_date","d1_date","d3_date","trough_date","r1_date","r3_date","p1_date","a1","a2","a3","b","c","d1_to_trough","d1_to_r3","d1_to_p1","p1_censored"]
    ext=ev.sort_values("b").head(25)[cols].copy()
    for c in ["peak_date","d1_date","d3_date","trough_date","r1_date","r3_date","p1_date"]:ext[c]=ext[c].dt.strftime("%Y-%m-%d")
    extreme=ext.where(pd.notna(ext),None).to_dict("records")
    duration_rows=[]
    for (idx,ma),g in ev.groupby(["index_key","ma_window"]):
        r={"index_name":name_map.get(idx,idx),"ma_window":int(ma),"n":int(len(g))}
        for col,label in DURATIONS:r[col]={**_duration_stats(g[col]),"label":label}
        r["p1_censored_count"]=int(g["p1_censored"].sum());r["p1_censored_rate"]=_f(g["p1_censored"].mean());duration_rows.append(r)
    bucket_defs=[(-np.inf,3,"≤3"),(3,5,"4–5"),(5,10,"6–10"),(10,20,"11–20"),(20,40,"21–40"),(40,60,"41–60"),(60,120,"61–120"),(120,np.inf,">120")]
    duration_buckets={}
    for col,label in DURATIONS:
        s=pd.to_numeric(ev[col],errors="coerce");denom=int(s.notna().sum());rows=[]
        for lo,hi,lab in bucket_defs:
            mask=(s<=hi) if lo==-np.inf else ((s>lo) if hi==np.inf else ((s>lo)&(s<=hi)))
            n=int(mask.sum());rows.append({"bucket":lab,"count":n,"rate":_f(n/denom) if denom else None})
        if col=="d1_to_p1":rows.append({"bucket":"未收复","count":int(ev.p1_censored.sum()),"rate":_f(ev.p1_censored.mean())})
        duration_buckets[col]={"label":label,"rows":rows}
    recovery_curves=[]
    for h in (3,5,10,20,40,60,120):
        recovery_curves.append({"days":h,"trough_rate":_f((ev.d1_to_trough<=h).mean()),"r3_rate":_f((ev.d1_to_r3<=h).mean()),"p1_rate":_f((pd.to_numeric(ev.d1_to_p1,errors="coerce")<=h).fillna(False).mean())})
    window_summary=[]
    if len(win):
        for h in (3,5,10,20,40,60):
            dd=pd.to_numeric(win.get(f"maxdd_{h}"),errors="coerce").dropna();rr=pd.to_numeric(win.get(f"ret_{h}"),errors="coerce").dropna()
            window_summary.append({"days":h,"n":int(len(dd)),"median_maxdd":_f(dd.median()),"p90_maxdd":_f(dd.quantile(.10)),"median_return":_f(rr.median()) if len(rr) else None,"p10_return":_f(rr.quantile(.10)) if len(rr) else None})
    ev["period"]=ev.d1_date.map(_assign_period);ca["period"]=ca.d1_date.map(_assign_period)
    period_summary=[];period_detail=[];signal_period=[]
    for name,label,start,end in PERIODS:
        g=ev[ev.period==name]
        if len(g):
            period_summary.append({"period":name,"label":label,"start":start,"end":cutoff if name=="2022至今" else end,"n":int(len(g)),"a1":_stats(g.a1),"a3":_stats(g.a3),"b":_stats(g.b),"c":_cstats(g.c),"duration":_duration_stats(g.d1_to_r3)})
            for (idx,ma),q in g.groupby(["index_key","ma_window"]):period_detail.append({"period":name,"label":label,"index_name":name_map.get(idx,idx),"ma_window":int(ma),"n":int(len(q)),"a1":_stats(q.a1),"a3":_stats(q.a3),"b":_stats(q.b),"c":_cstats(q.c),"duration":_duration_stats(q.d1_to_r3)})
        cg=ca[ca.period==name]
        if len(cg):
            d1=len(cg);d2=int(cg.reached_day2.sum());true=int(cg.true_breakdown.sum());signal_period.append({"period":name,"day1_candidates":d1,"day2_reached":d2,"true_breakdowns":true,"day1_success_rate":_f(true/d1),"day2_success_rate":_f(true/d2) if d2 else None})
    conduction_top=[]
    if len(con):
        con=con[pd.to_numeric(con.denominator,errors="coerce")>=20].copy()
        for (ma,h),g in con.groupby(["ma_window","horizon"]):
            for _,r in g.sort_values("probability",ascending=False).head(6).iterrows():conduction_top.append({"ma_window":int(ma),"horizon":int(h),"source_name":r.get("source_name",name_map.get(r.source,r.source)),"target_name":r.get("target_name",name_map.get(r.target,r.target)),"denominator":int(r.denominator),"hits":int(r.hits),"probability":_f(r.probability)})
    return {"signal_reliability":sig,"signal_period":signal_period,"depth_tables":depth,"ma_summaries":ma_summ,"style_comparison":style,"extreme_events":extreme,"duration_rows":duration_rows,"duration_buckets":duration_buckets,"recovery_curves":recovery_curves,"window_summary":window_summary,"period_summary":period_summary,"period_detail":period_detail,"conduction_top":conduction_top}

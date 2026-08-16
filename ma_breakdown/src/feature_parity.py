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
("2022至今","价值回归+小微盘","2022-01-01","2026-07-31")]


def _f(v):
    return None if v is None or pd.isna(v) else float(v)


def _stats(s):
    s=pd.to_numeric(pd.Series(s),errors="coerce").dropna()
    if s.empty:return {"n":0}
    return {"n":int(len(s)),"mean":_f(s.mean()),"median":_f(s.median()),"std":_f(s.std(ddof=1)),
            "p25":_f(s.quantile(.25)),"p75":_f(s.quantile(.75)),"p75_risk":_f(s.quantile(.25)),
            "p90_risk":_f(s.quantile(.10)),"p95_risk":_f(s.quantile(.05)),"p99_risk":_f(s.quantile(.01)),
            "extreme":_f(s.min()),"best":_f(s.max()),"prob_gt_3":_f((s<=-.03).mean()),
            "prob_gt_5":_f((s<=-.05).mean()),"prob_gt_10":_f((s<=-.10).mean()),
            "prob_gt_15":_f((s<=-.15).mean()),"prob_gt_20":_f((s<=-.20).mean())}


def _cstats(s):
    s=pd.to_numeric(pd.Series(s),errors="coerce").dropna()
    if s.empty:return {"n":0}
    return {"n":int(len(s)),"mean":_f(s.mean()),"median":_f(s.median()),"std":_f(s.std(ddof=1)),
            "p10":_f(s.quantile(.10)),"p25":_f(s.quantile(.25)),"p75":_f(s.quantile(.75)),"p90":_f(s.quantile(.90)),
            "worst":_f(s.min()),"best":_f(s.max()),"positive_rate":_f((s>0).mean()),"loss_rate":_f((s<0).mean())}


def _dstats(s):
    s=pd.to_numeric(pd.Series(s),errors="coerce").dropna()
    if s.empty:return {"n":0}
    return {"n":int(len(s)),"mean":_f(s.mean()),"median":_f(s.median()),"p75":_f(s.quantile(.75)),
            "p90":_f(s.quantile(.90)),"p95":_f(s.quantile(.95)),"max":_f(s.max()),
            "le5":_f((s<=5).mean()),"le10":_f((s<=10).mean()),"le20":_f((s<=20).mean()),"gt40":_f((s>40).mean())}


def _risk(g):
    if g.empty:return {"n":0}
    return {"n":int(len(g)),"a1_mean":_f(g.a1.mean()),"a1_median":_f(g.a1.median()),"a1_p90":_f(g.a1.quantile(.10)),
            "a3_mean":_f(g.a3.mean()),"a3_median":_f(g.a3.median()),"a3_p90":_f(g.a3.quantile(.10)),
            "b_mean":_f(g.b.mean()),"b_median":_f(g.b.median()),"c_median":_f(g.c.median()),"c_loss_rate":_f((g.c<0).mean()),
            "prob_a1_5":_f((g.a1<=-.05).mean()),"prob_a1_10":_f((g.a1<=-.10).mean()),
            "prob_a3_5":_f((g.a3<=-.05).mean()),"prob_a3_10":_f((g.a3<=-.10).mean()),
            "d3_to_trough_median":_f(g.d3_to_trough.median()),"d1_to_r3_median":_f(g.d1_to_r3.median())}


def _period(d):
    if pd.isna(d):return None
    d=pd.Timestamp(d)
    for name,_,start,end in PERIODS:
        if pd.Timestamp(start)<=d<=pd.Timestamp(end):return name
    return None


def _bucket(ev,feature,bins,labels):
    rows=[]
    for ma in (20,25):
        x=ev[ev.ma_window==ma].copy();x["bucket"]=pd.cut(pd.to_numeric(x[feature],errors="coerce"),bins=bins,labels=labels,include_lowest=True)
        for label in labels:
            g=x[x.bucket==label]
            if g.empty:continue
            r={"ma_window":ma,"bucket":str(label),"feature_median":_f(pd.to_numeric(g[feature],errors="coerce").median())};r.update(_risk(g));rows.append(r)
    return rows


def _long_trend(ev):
    cols=["below_ma60","below_ma120","below_ma200"]
    if any(c not in ev for c in cols):return []
    x=ev.dropna(subset=cols).copy()
    if x.empty:return []
    x["k"]=x[cols].astype(int).sum(axis=1);labels=["0/3：仅短周期跌破","1/3：部分中期转弱","2/3：中长期同步转弱","3/3：MA60/120/200全在下方"]
    rows=[]
    for ma in (20,25):
        for k,label in enumerate(labels):
            g=x[(x.ma_window==ma)&(x.k==k)]
            if g.empty:continue
            r={"ma_window":ma,"state":label};r.update(_risk(g));rows.append(r)
    return rows


def _escalation(ev):
    rows=[]
    for idx,g in ev.groupby("index_key"):
        d25=pd.to_datetime(g.loc[g.ma_window==25,"d3_date"])
        for _,r in g[g.ma_window==20].iterrows():
            rows.append({**r.to_dict(),"escalated":bool(((d25>=pd.Timestamp(r.d3_date))&(d25<=pd.Timestamp(r.r3_date))).any())})
    if not rows:return []
    x=pd.DataFrame(rows);out=[]
    for state,g in x.groupby("escalated"):
        r={"state":"MA20周期内进一步确认MA25真跌破" if state else "未进一步确认MA25真跌破"};r.update(_risk(g));out.append(r)
    return out


def _unrecovered(ev):
    rows=[]
    for ma in (20,25):
        g0=ev[ev.ma_window==ma].copy();dur=pd.to_numeric(g0.d1_to_r3,errors="coerce")
        for elapsed in (5,10,20,40):
            g=g0[dur>elapsed].copy()
            if g.empty:continue
            rem=pd.to_numeric(g.d1_to_r3,errors="coerce")-elapsed
            r={"ma_window":ma,"elapsed":elapsed,"risk_n":int(len(g)),"recover_next5_rate":_f((rem<=5).mean()),"remaining_median":_f(rem.median())};r.update(_risk(g));rows.append(r)
    return rows


def _corr(a,b):
    x=pd.DataFrame({"a":pd.to_numeric(a,errors="coerce"),"b":pd.to_numeric(b,errors="coerce")}).dropna()
    return None if len(x)<3 or x.a.nunique()<2 or x.b.nunique()<2 else _f(x.a.corr(x.b))


def _enhanced_conduction(ev,frames,name_map):
    rows=[]
    if ev.empty or not frames:return rows
    dates={k:pd.DatetimeIndex(pd.to_datetime(f.date)).sort_values().unique() for k,f in frames.items()}
    ranges={k:(d[0],d[-1]) for k,d in dates.items() if len(d)}
    positions={k:{pd.Timestamp(d):i for i,d in enumerate(di)} for k,di in dates.items()}
    for ma in (20,25):
        z=ev[ev.ma_window==ma]
        for source in frames:
            src=pd.to_datetime(z.loc[z.index_key==source,"d3_date"])
            for target in frames:
                if source==target:continue
                td=dates[target];targets=sorted(pd.to_datetime(z.loc[z.index_key==target,"d3_date"]).tolist())
                for h in (3,5,10,20):
                    denom=hits=same=0;lags=[]
                    for d in src:
                        d=pd.Timestamp(d)
                        if target not in ranges or not(ranges[target][0]<=d<=ranges[target][1]):continue
                        pos=int(td.searchsorted(d,side="left"));endpos=pos+h
                        if pos>=len(td) or endpos>=len(td):continue
                        denom+=1;end=td[endpos];cand=[x for x in targets if d<=x<=end]
                        if not cand:continue
                        hit=pd.Timestamp(cand[0]);hits+=1;hp=positions[target].get(hit)
                        if hp is not None:
                            lag=int(hp-pos);lags.append(lag);same+=int(lag==0)
                    rows.append({"source_name":name_map[source],"target_name":name_map[target],"source":source,"target":target,
                                 "ma_window":ma,"horizon":h,"denominator":denom,"hits":hits,"probability":_f(hits/denom) if denom else None,
                                 "same_day_rate":_f(same/denom) if denom else None,"avg_lag":_f(np.mean(lags)) if lags else None,"median_lag":_f(np.median(lags)) if lags else None})
    return rows


def _conduction_wide(rows):
    if not rows:return [],{"20":[],"25":[]}
    x=pd.DataFrame(rows);wide=[]
    for (s,t,ma),g in x.groupby(["source_name","target_name","ma_window"]):
        rec={"source_name":s,"target_name":t,"ma_window":int(ma)}
        for _,r in g.iterrows():
            h=int(r.horizon);rec[f"rate_{h}"]=_f(r.probability);rec[f"n_{h}"]=int(r.denominator)
            if h==10:rec.update({"denominator":int(r.denominator),"same_day_rate":_f(r.same_day_rate),"avg_lag_10":_f(r.avg_lag),"median_lag_10":_f(r.median_lag)})
        wide.append(rec)
    names=sorted(set(x.source_name)|set(x.target_name));mats={"20":[],"25":[]}
    for ma in (20,25):
        for s in names:
            cells=[]
            for t in names:
                if s==t:cells.append({"target_name":t,"diag":True});continue
                r=next((q for q in wide if q["ma_window"]==ma and q["source_name"]==s and q["target_name"]==t),None)
                cells.append({"target_name":t,**(r or {})})
            mats[str(ma)].append({"source_name":s,"cells":cells})
    return wide,mats


def _validation(frames,instruments,ev):
    rows=[]
    for ins in instruments:
        f=frames.get(ins["key"])
        if f is None or f.empty:continue
        g=ev[ev.index_key==ins["key"]]
        if g.empty:continue
        idx=int(g.iloc[len(g)//2].d3_idx)
        for ma in (20,25):
            if idx+1<ma:continue
            manual=float(pd.to_numeric(f.loc[idx-ma+1:idx,"close"]).mean());rolling=float(f.at[idx,f"ma{ma}"])
            rows.append({"index_name":ins["name"],"date":str(pd.Timestamp(f.at[idx,"date"]).date()),"ma_window":ma,
                         "manual":manual,"rolling":rolling,"abs_diff":abs(manual-rolling),"match":abs(manual-rolling)<1e-10})
    return rows


def _signal_chart(frames,ev,ongoing,instruments):
    prices={}
    for ins in instruments:
        f=frames.get(ins["key"])
        if f is None:continue
        q=f[pd.to_datetime(f.date)>=pd.Timestamp("2005-01-01")]
        prices[ins["key"]]=[[str(pd.Timestamp(r.date).date()),_f(r.close),_f(r.ma20),_f(r.ma25)] for r in q.itertuples()]
    markers=[]
    for r in ev.itertuples():
        markers.append({"index_key":r.index_key,"index_name":r.index_name,"ma_window":int(r.ma_window),"d1_date":str(pd.Timestamp(r.d1_date).date()),
                        "d3_date":str(pd.Timestamp(r.d3_date).date()),"r1_date":str(pd.Timestamp(r.r1_date).date()),"r3_date":str(pd.Timestamp(r.r3_date).date()),"completed":True})
    if ongoing is not None and len(ongoing):
        for r in ongoing.itertuples():
            markers.append({"index_key":r.index_key,"index_name":r.index_name,"ma_window":int(r.ma_window),"d1_date":str(pd.Timestamp(r.d1_date).date()),
                            "d3_date":str(pd.Timestamp(r.d3_date).date()),"r1_date":None,"r3_date":None,"completed":False})
    return {"periods":[{"name":n,"label":l,"start":s,"end":e} for n,l,s,e in PERIODS],
            "indices":[{"key":x["key"],"name":x["name"]} for x in instruments],"prices":prices,"events":markers}


def enhance_research_payload(events,candidates,instruments,cutoff,frames=None,ongoing=None):
    ev=events.copy();ca=candidates.copy();frames=frames or {};name_map={x["key"]:x["name"] for x in instruments}
    for c in ["peak_date","d1_date","d2_date","d3_date","trough_date","r1_date","r3_date","p1_date"]:
        if c in ev:ev[c]=pd.to_datetime(ev[c],errors="coerce")
    if "d1_date" in ca:ca["d1_date"]=pd.to_datetime(ca.d1_date,errors="coerce")
    depth=[]
    for (idx,ma),g in ev.groupby(["index_key","ma_window"]):
        depth.append({"index_name":name_map[idx],"index_key":idx,"ma_window":int(ma),"n":int(len(g)),
                      "metrics":{m:_stats(g[m]) for m in ("a1","a2","a3","b")},"c":_cstats(g.c),
                      "d1_to_trough":_dstats(g.d1_to_trough),"d3_to_trough":_dstats(g.d3_to_trough),
                      "d1_to_r3":_dstats(g.d1_to_r3),"d1_to_p1":_dstats(g.d1_to_p1)})
    duration_risk=[]
    for ma in (20,25):
        z=ev[ev.ma_window==ma];s=pd.to_numeric(z.d1_to_r3,errors="coerce")
        for lo,hi,label in [(-np.inf,5,"≤5"),(5,10,"6–10"),(10,20,"11–20"),(20,40,"21–40"),(40,np.inf,">40")]:
            mask=(s<=hi) if lo==-np.inf else ((s>lo) if hi==np.inf else ((s>lo)&(s<=hi)));g=z[mask]
            if g.empty:continue
            duration_risk.append({"ma_window":ma,"bucket":label,**_risk(g)})
    correlations=[]
    for (idx,ma),g in ev.groupby(["index_key","ma_window"]):
        correlations.append({"index_name":name_map[idx],"ma_window":int(ma),"n":int(len(g)),"corr_a1":_corr(g.d1_to_r3,-g.a1),
                             "corr_a3":_corr(g.d1_to_r3,-g.a3),"corr_b":_corr(g.d1_to_r3,-g.b),"corr_p1_b":_corr(g.d1_to_p1,-g.b)})
    long_cols=[c for c in ["index_name","ma_window","peak_date","d1_date","d3_date","trough_date","r3_date","p1_date","a1","a3","b","c","d1_to_r3","d1_to_p1"] if c in ev]
    longest=ev.sort_values("d1_to_r3",ascending=False).head(30)[long_cols].copy()
    for c in ["peak_date","d1_date","d3_date","trough_date","r3_date","p1_date"]:
        if c in longest:longest[c]=pd.to_datetime(longest[c],errors="coerce").dt.strftime("%Y-%m-%d")
    conditional_overall=[]
    for ma in (20,25):
        g=ev[ev.ma_window==ma];s=g.a1
        conditional_overall.append({"ma_window":ma,"n":int(len(g)),"prob_a1_5":_f((s<=-.05).mean()),"prob_a1_10":_f((s<=-.10).mean()),
                                    "prob_a1_15":_f((s<=-.15).mean()),"prob_a1_20":_f((s<=-.20).mean()),"prob_a3_10":_f((g.a3<=-.10).mean()),
                                    "prob_r3_gt20":_f((g.d1_to_r3>20).mean()),"prob_r3_gt40":_f((g.d1_to_r3>40).mean()),"c_loss_rate":_f((g.c<0).mean()),
                                    "worst10_mean":_f(s[s<=s.quantile(.10)].mean()),"worst5_mean":_f(s[s<=s.quantile(.05)].mean())})
    blocks=[
      {"key":"d3_dev","title":"Day3偏离均线深度","rows":_bucket(ev,"d3_ma_dev",[-np.inf,-.04,-.02,-.01,0],["≤-4%","-4%~-2%","-2%~-1%",">-1%"])},
      {"key":"d1_d3","title":"D1→D3跌幅","rows":_bucket(ev,"d1_to_d3",[-np.inf,-.04,-.02,-.01,np.inf],["≤-4%","-4%~-2%","-2%~-1%",">-1%"])},
      {"key":"breadth","title":"多指数共振广度","rows":_bucket(ev,"breadth_ratio",[-np.inf,.30,.50,.70,np.inf],["≤30%","30%~50%","50%~70%",">70%"])},
      {"key":"long_trend","title":"长期趋势背景","rows":_long_trend(ev)},
      {"key":"escalation","title":"MA20→MA25升级","rows":_escalation(ev)},
      {"key":"unrecovered","title":"已持续未收回","rows":_unrecovered(ev)}]
    ev["period"]=ev.d1_date.map(_period);ca["period"]=ca.d1_date.map(_period)
    pma=[];pdetail=[]
    for name,label,start,end in PERIODS:
        effective=cutoff if name=="2022至今" else end;g=ev[ev.period==name]
        for ma in (20,25):
            q=g[g.ma_window==ma]
            if not q.empty:pma.append({"period":name,"label":label,"start":start,"end":effective,"ma_window":ma,"n":int(len(q)),
                                       "a1":_stats(q.a1),"a2":_stats(q.a2),"a3":_stats(q.a3),"b":_stats(q.b),"c":_cstats(q.c),
                                       "duration":_dstats(q.d1_to_r3),"p1":_dstats(q.d1_to_p1)})
        for (idx,ma),q in g.groupby(["index_key","ma_window"]):
            pdetail.append({"period":name,"label":label,"index_name":name_map[idx],"ma_window":int(ma),"n":int(len(q)),
                            "a1":_stats(q.a1),"a2":_stats(q.a2),"a3":_stats(q.a3),"b":_stats(q.b),"c":_cstats(q.c),
                            "duration":_dstats(q.d1_to_r3),"p1":_dstats(q.d1_to_p1)})
    cond=_enhanced_conduction(ev,frames,name_map) if frames else [];wide,mats=_conduction_wide(cond)
    return {"depth_tables":depth,"duration_risk_buckets":duration_risk,"duration_correlations":correlations,
            "longest_events":longest.where(pd.notna(longest),None).to_dict("records"),"conditional_overall":conditional_overall,
            "conditional_blocks":blocks,"period_ma_summary":pma,"period_detail":pdetail,"conduction_detail":cond,"conduction_wide":wide,
            "conduction_matrices":mats,"ma_validation":_validation(frames,instruments,ev) if frames else [],
            "signal_chart":_signal_chart(frames,ev,ongoing,instruments) if frames else {"prices":{},"events":[],"periods":[],"indices":[]}}

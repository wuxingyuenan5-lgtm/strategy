from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from ma_breakdown.src.data import load_instruments, fetch_tencent_instrument
from ma_breakdown.src.events import compute_ma, detect_events, detect_candidate_records
from ma_breakdown.src.metrics import enrich_event_metrics, depth_summary, duration_summary, window_path_stats
from ma_breakdown.src.analytics import add_risk_features, add_breadth, select_conditional_panels, conduction_matrix
from ma_breakdown.src.charts import line_svg, event_path_svg
from ma_breakdown.src.render import render_report
from ma_breakdown.src.validate import validate_payload
from ma_breakdown.src.report_data import build_research_payload
from ma_breakdown.src.narrative import build_narrative

FEATURE_LABELS={
    'd3_ma_dev':'D3相对均线偏离','d1_to_d3':'D1→D3跌幅','peak_to_d1':'Peak→D1回撤',
    'peak_to_d3':'Peak→D3回撤','ma_slope_5':'均线5日斜率','ret20':'前20日收益','ret60':'前60日收益',
    'rv20':'20日实现波动率','rv20_pct':'波动率历史分位','breadth_ratio':'同期真跌破广度'
}
METRIC_LABELS={'a1':'甲1','a2':'甲2','a3':'甲3','b':'乙'}

def _iso(v):
    if v is None:return None
    if isinstance(v,(pd.Timestamp,np.datetime64)):return None if pd.isna(v) else str(pd.Timestamp(v).date())
    if isinstance(v,(np.floating,float)):return None if pd.isna(v) else float(v)
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.bool_,)):return bool(v)
    return v

def _records(df:pd.DataFrame)->list[dict]:
    return [{k:_iso(v) for k,v in r.items()} for r in df.to_dict('records')]

def _dump(path:Path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=_iso),encoding='utf-8')

def _quality(frames,instruments,cutoff,strict_cutoff=True):
    checks=[];severity='PASS';cutoff_ts=pd.Timestamp(cutoff)
    def add(status,name,message):
        nonlocal severity
        checks.append({'status':status,'name':name,'message':message})
        if status=='FAIL':severity='FAIL'
        elif status=='WARN' and severity=='PASS':severity='WARN'
    for ins in instruments:
        key=ins['key'];f=frames[key].sort_values('date').reset_index(drop=True)
        if f.empty:
            add('FAIL',f'{ins["name"]} 数据','无历史数据');continue
        if f['date'].duplicated().any():add('FAIL',f'{ins["name"]} 日期','存在重复交易日')
        else:add('PASS',f'{ins["name"]} 日期','无重复交易日')
        if not f['date'].is_monotonic_increasing:add('FAIL',f'{ins["name"]} 排序','日期非升序')
        last=pd.Timestamp(f['date'].max())
        if strict_cutoff and last!=cutoff_ts:add('FAIL',f'{ins["name"]} 截止日',f'最新={last.date()}，目标={cutoff}')
        else:add('PASS',f'{ins["name"]} 截止日',f'最新={last.date()}')
        actual=pd.Timestamp(f['date'].min());desired=pd.Timestamp(ins['sample_start'])
        if actual>desired:add('WARN',f'{ins["name"]} 样本起点',f'连续数据起点 {actual.date()} 晚于目标 {desired.date()}，按实际可用起点统计')
        else:add('PASS',f'{ins["name"]} 样本起点',f'数据起点={actual.date()}，目标事件起点={desired.date()}')
        if len(f)<80:add('FAIL',f'{ins["name"]} 记录数',f'仅{len(f)}条')
    return {'status':severity,'checks':checks}

def _period_rows(candidates:pd.DataFrame,events:pd.DataFrame):
    if candidates.empty:return []
    c=candidates.copy();c['year']=pd.to_datetime(c['d1_date']).dt.year;rows=[]
    for (year,ma),g in c.groupby(['year','ma_window']):
        d1=len(g);d2=int(g['reached_day2'].sum());true=int(g['true_breakdown'].sum())
        evn=int(((pd.to_datetime(events['d1_date']).dt.year==year)&(events['ma_window']==ma)).sum()) if not events.empty else 0
        rows.append({'year':int(year),'ma_window':int(ma),'events':evn,'day1_candidates':d1,'day2_reached':d2,'true_breakdowns':true,'day1_success_rate':true/d1 if d1 else None,'day2_success_rate':true/d2 if d2 else None})
    return sorted(rows,key=lambda r:(r['year'],r['ma_window']))

def _build_charts(events,period_rows,window_rows,frames):
    years=sorted({r['year'] for r in period_rows});period_series={}
    for ma in (20,25):
        m={r['year']:r['events'] for r in period_rows if r['ma_window']==ma};period_series[f'MA{ma}真跌破']=[(y,float(m.get(y,0))) for y in years]
    period_svg=line_svg(period_series);horizons=[3,5,10,20,40,60,120]
    duration_svg=line_svg({'已见底':[(h,float((pd.to_numeric(events['d1_to_trough'],errors='coerce')<=h).mean())) for h in horizons] if len(events) else [],'已真收回R3':[(h,float((pd.to_numeric(events['d1_to_r3'],errors='coerce')<=h).mean())) for h in horizons] if len(events) else [],'已收复D1价格':[(h,float((pd.to_numeric(events['d1_to_p1'],errors='coerce')<=h).fillna(False).mean())) for h in horizons] if len(events) else []},y_percent=True)
    wh=[3,5,10,20,40,60];window_series={'中位最大跌幅':[],'P90最大跌幅':[],'中位期末收益':[]}
    if not window_rows.empty:
        for h in wh:
            dd=pd.to_numeric(window_rows[f'maxdd_{h}'],errors='coerce').dropna();rr=pd.to_numeric(window_rows[f'ret_{h}'],errors='coerce').dropna()
            window_series['中位最大跌幅'].append((h,float(dd.median()) if len(dd) else np.nan));window_series['P90最大跌幅'].append((h,float(dd.quantile(.10)) if len(dd) else np.nan));window_series['中位期末收益'].append((h,float(rr.median()) if len(rr) else np.nan))
    window_svg=line_svg(window_series,y_percent=True);event_svg=line_svg({})
    if len(events):
        e=events.loc[pd.to_numeric(events['b'],errors='coerce').idxmin()];event_svg=event_path_svg(frames[e['index_key']],e)
    return {'period_svg':period_svg,'duration_svg':duration_svg,'window_svg':window_svg,'event_svg':event_svg}

def run_from_frames(frames:dict[str,pd.DataFrame],instruments:list[dict],cutoff:str,output_dir:str|Path,template_path:str|Path,version='v0.2',strict_cutoff=True,check_regression_anchors=False):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);(out/'raw').mkdir(exist_ok=True);cutoff_ts=pd.Timestamp(cutoff)
    prepared={};inst_rows=[];raw_manifest=[]
    for ins in instruments:
        source_meta=frames[ins['key']].attrs.get('source_meta',{});f=frames[ins['key']].copy();f['date']=pd.to_datetime(f['date']);f=f[f['date']<=cutoff_ts].sort_values('date').drop_duplicates('date').reset_index(drop=True)
        for w in (20,25,60,120,200):f[f'ma{w}']=compute_ma(f,w)
        prepared[ins['key']]=f;f.to_csv(out/'raw'/f'{ins["key"]}.csv',index=False);actual_start=str(f['date'].min().date()) if len(f) else None;effective=max(pd.Timestamp(ins['sample_start']),f['date'].min()) if len(f) else pd.Timestamp(ins['sample_start'])
        inst_rows.append({**ins,'actual_start':actual_start,'sample_start_effective':str(effective.date()),'rows':int(len(f))});raw_manifest.append({'key':ins['key'],'name':ins['name'],'rows':int(len(f)),'first':actual_start,'last':str(f['date'].max().date()) if len(f) else None,'source':source_meta.get('source','injected'),'source_url':source_meta.get('url'),'source_symbol':source_meta.get('symbol',ins.get('tencent_symbol'))})
    quality=_quality(prepared,instruments,cutoff,strict_cutoff=strict_cutoff)
    if quality['status']=='FAIL':_dump(out/'data_quality.json',quality);raise ValueError('data quality failed before analysis')
    completed=[];ongoing=[];candidates=[];windows=[];name_map={x['key']:x['name'] for x in instruments};availability={k:pd.DatetimeIndex(v['date']) for k,v in prepared.items()}
    for ins in instruments:
        p=prepared[ins['key']];sample_start=str(max(pd.Timestamp(ins['sample_start']),p['date'].min()).date())
        for ma in (20,25):
            ma_col=f'ma{ma}';ev,on=detect_events(p,ma_col,sample_start,cutoff,peak_lookback=10)
            if len(ev):
                ev['index_key']=ins['key'];ev['index_name']=ins['name'];ev['ma_window']=ma;ev=enrich_event_metrics(ev,p);ev=add_risk_features(ev,p,ma);completed.append(ev);wr=window_path_stats(ev,p);wr['index_key']=ins['key'];wr['ma_window']=ma;windows.append(wr)
            if len(on):on['index_key']=ins['key'];on['index_name']=ins['name'];on['ma_window']=ma;ongoing.append(on)
            cr=detect_candidate_records(p,ma_col,sample_start,cutoff)
            if len(cr):cr['index_key']=ins['key'];cr['index_name']=ins['name'];cr['ma_window']=ma;candidates.append(cr)
    events=pd.concat(completed,ignore_index=True) if completed else pd.DataFrame();ongoing_df=pd.concat(ongoing,ignore_index=True) if ongoing else pd.DataFrame();cand=pd.concat(candidates,ignore_index=True) if candidates else pd.DataFrame(columns=['d1_date','reached_day2','true_breakdown','ma_window','index_key']);win=pd.concat(windows,ignore_index=True) if windows else pd.DataFrame()
    if len(events):events=add_breadth(events,availability)
    cond=[]
    for ma in (20,25):
        for h in (5,10,20):
            cm=conduction_matrix(events,availability,horizon=h,ma_window=ma)
            if len(cm):cond.append(cm)
    conduction=pd.concat(cond,ignore_index=True) if cond else pd.DataFrame()
    if len(conduction):conduction['source_name']=conduction['source'].map(name_map);conduction['target_name']=conduction['target'].map(name_map)
    depth=[]
    if len(events):
        for (key,ma),g in events.groupby(['index_key','ma_window']):
            ds=depth_summary(g)
            for _,r in ds.iterrows():depth.append({'index_key':key,'index_name':name_map[key],'ma_window':int(ma),**r.to_dict(),'metric_label':METRIC_LABELS.get(r['metric'],r['metric'])})
    duration=duration_summary(events) if len(events) else {};features=['d3_ma_dev','d1_to_d3','peak_to_d1','peak_to_d3','ma_slope_5','ret20','ret60','rv20','rv20_pct','breadth_ratio'];min_bucket=max(8,min(20,len(events)//40)) if len(events) else 8;panels=select_conditional_panels(events,features,max_panels=4,min_bucket=min_bucket) if len(events) else []
    for p in panels:p['feature_label']=FEATURE_LABELS.get(p['feature'],p['feature'])
    period_rows=_period_rows(cand,events);charts=_build_charts(events,period_rows,win,prepared);all_a1=pd.to_numeric(events.get('a1',pd.Series(dtype=float)),errors='coerce').dropna();all_c=pd.to_numeric(events.get('c',pd.Series(dtype=float)),errors='coerce').dropna();d1_n=len(cand);d2_n=int(cand['reached_day2'].sum()) if d1_n else 0;true_n=int(cand['true_breakdown'].sum()) if d1_n else 0
    overview={'median_a1':float(all_a1.median()) if len(all_a1) else None,'p90_a1':float(all_a1.quantile(.10)) if len(all_a1) else None,'median_c':float(all_c.median()) if len(all_c) else None,'day1_success_rate':true_n/d1_n if d1_n else None,'day2_success_rate':true_n/d2_n if d2_n else None}
    event_records=_records(events.sort_values(['d1_date','index_key','ma_window'],ascending=[False,True,True])) if len(events) else []
    research=build_research_payload(events,cand,win,conduction,instruments,cutoff);narrative=build_narrative(research,panels)
    payload={'meta':{'version':version,'cutoff':cutoff,'event_count':len(event_records),'ongoing_count':int(len(ongoing_df))},'instruments':inst_rows,'overview':overview,'depth_rows':depth,'conditional_panels':panels,'duration':duration,'period_rows':period_rows,'conduction_rows':_records(conduction) if len(conduction) else [],'events':event_records,'charts':charts,'data_quality':quality,'research':research,'narrative':narrative}
    errors=validate_payload(payload)
    if errors:quality['status']='FAIL';quality['checks'].append({'status':'FAIL','name':'payload','message':'; '.join(errors)});_dump(out/'data_quality.json',quality);raise ValueError('payload validation failed: '+'; '.join(errors))
    render_report(payload,template_path,out/'index.html');events.to_csv(out/'events.csv',index=False);cand.to_csv(out/'candidate_signals.csv',index=False);_dump(out/'raw_manifest.json',raw_manifest);_dump(out/'data_quality.json',quality)
    summary={'meta':payload['meta'],'instrument_names':[x['name'] for x in inst_rows],'overview':overview,'depth_rows':_records(pd.DataFrame(depth)) if depth else [],'duration':duration,'conditional_panels':panels,'research':research,'narrative':narrative};_dump(out/'summary.json',summary)
    if len(conduction):conduction.to_csv(out/'conduction.csv',index=False)
    if len(ongoing_df):ongoing_df.to_csv(out/'ongoing_events.csv',index=False)
    return payload

def run_network(config_dir:Path,cutoff:str,output_dir:Path,template_path:Path):
    instruments=load_instruments(config_dir/'instruments.yaml');cfg=yaml.safe_load((config_dir/'research.yaml').read_text(encoding='utf-8'));begin=str(cfg.get('warmup_begin','2004-01-01'));end=cutoff;frames={}
    for ins in instruments:
        df,meta=fetch_tencent_instrument(ins,begin,end,timeout=20,retries=3,window_years=2);df.attrs['source_meta']=meta;frames[ins['key']]=df
    return run_from_frames(frames,instruments,cutoff,output_dir,template_path,version='v0.2',strict_cutoff=True,check_regression_anchors=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cutoff',default='2026-07-31');ap.add_argument('--output',default='ma_breakdown/output/v0.2');ap.add_argument('--config-dir',default='ma_breakdown/config');ap.add_argument('--template',default='ma_breakdown/templates/report.html');args=ap.parse_args();run_network(Path(args.config_dir),args.cutoff,Path(args.output),Path(args.template))
if __name__=='__main__':main()

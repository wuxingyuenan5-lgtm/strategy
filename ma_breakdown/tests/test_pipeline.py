import json
from pathlib import Path
import numpy as np
import pandas as pd

from ma_breakdown.run_pipeline import run_from_frames


def synthetic_frame(start="2004-01-01", end="2026-07-31", phase=0.0):
    dates=pd.bdate_range(start, end)
    t=np.arange(len(dates),dtype=float)
    close=100 + 9*np.sin((t+phase)/18.0) + 2*np.sin((t+phase)/5.0) + 0.015*t
    return pd.DataFrame({"date":dates,"open":close,"high":close,"low":close,"close":close,"volume":1.0,"amount":1.0})


def instrument_config():
    return [
        {"key":"sse","name":"上证指数","code":"000001","secid":"1.000001","category":"宽基综合","sample_start":"2005-01-04"},
        {"key":"sse50","name":"上证50","code":"000016","secid":"1.000016","category":"大盘蓝筹","sample_start":"2005-01-04"},
        {"key":"csi300","name":"沪深300","code":"000300","secid":"1.000300","category":"大盘蓝筹","sample_start":"2005-01-04"},
        {"key":"csi1000","name":"中证1000","code":"000852","secid":"1.000852","category":"小盘","sample_start":"2005-01-04"},
        {"key":"chinext","name":"创业板指","code":"399006","secid":"0.399006","category":"创业成长","sample_start":"2005-01-04"},
        {"key":"star50","name":"科创50","code":"000688","secid":"1.000688","category":"科创成长","sample_start":"2005-01-04"},
        {"key":"dividend","name":"中证红利","code":"000922","secid":"1.000922","category":"防守红利","sample_start":"2005-01-04"},
    ]


def test_end_to_end_fixture_pipeline_writes_auditable_bundle(tmp_path: Path):
    inst=instrument_config()
    frames={x["key"]:synthetic_frame(phase=i*3) for i,x in enumerate(inst)}
    cutoff="2026-07-31"
    out=tmp_path/"v0.2"
    payload=run_from_frames(
        frames=frames,
        instruments=inst,
        cutoff=cutoff,
        output_dir=out,
        template_path=Path(__file__).parents[1]/"templates"/"report.html",
        version="v0.2-test",
        strict_cutoff=False,
    )
    for name in ["index.html","events.csv","candidate_signals.csv","summary.json","data_quality.json","raw_manifest.json"]:
        assert (out/name).exists(), name
    summary=json.loads((out/"summary.json").read_text())
    quality=json.loads((out/"data_quality.json").read_text())
    assert summary["meta"]["event_count"] == len(payload["events"])
    assert quality["status"] in {"PASS","WARN"}
    assert set(summary["instrument_names"]) == {x["name"] for x in inst}
    html=(out/"index.html").read_text()
    assert "研究概览" in html and "事件明细" in html
    assert "上证50" in html and "中证全指" not in html
    assert "甲1" in html and "D1→P1" in html


def test_run_network_uses_tencent_fetcher(monkeypatch, tmp_path):
    import ma_breakdown.run_pipeline as rp
    calls=[]
    def fake_fetch(ins, begin, end, **kwargs):
        calls.append((ins['key'],begin,end,kwargs))
        df=pd.DataFrame({'date':[pd.Timestamp('2026-07-31')],'close':[100.0]})
        return df, {'source':'tencent','symbol':ins.get('tencent_symbol')}
    monkeypatch.setattr(rp,'fetch_tencent_instrument',fake_fetch)
    monkeypatch.setattr(rp,'run_from_frames',lambda frames,instruments,cutoff,output_dir,template_path,**kwargs: {'keys':sorted(frames),'cutoff':cutoff})
    result=rp.run_network(Path('ma_breakdown/config'),'2026-07-31',tmp_path/'out',Path('ma_breakdown/templates/report.html'))
    assert len(calls)==7
    assert all(c[3]['window_years']==2 for c in calls)
    assert result['cutoff']=='2026-07-31'

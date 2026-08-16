import pandas as pd
import pytest
from ma_breakdown.src.data import parse_eastmoney_klines

INSTRUMENT = {"key":"sse","name":"上证指数","code":"000001","secid":"1.000001"}

def payload(lines, name="上证指数", code="000001"):
    return {"data":{"code":code,"name":name,"klines":lines}}

def test_parse_maps_close_and_orders_dates():
    p = payload([
        "2026-01-05,10,11,12,9,100,1000,0,0,0,0",
        "2026-01-02,9,10,11,8,90,900,0,0,0,0",
    ])
    df = parse_eastmoney_klines(p, INSTRUMENT, cutoff="2026-01-05")
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2026-01-02","2026-01-05"]
    assert list(df["close"]) == [10.0,11.0]

def test_parse_rejects_duplicate_dates():
    p = payload([
        "2026-01-02,9,10,11,8,90,900,0,0,0,0",
        "2026-01-02,9,10,11,8,90,900,0,0,0,0",
    ])
    with pytest.raises(ValueError, match="duplicate"):
        parse_eastmoney_klines(p, INSTRUMENT)

def test_parse_clips_cutoff():
    p = payload([
        "2026-01-02,9,10,11,8,90,900,0,0,0,0",
        "2026-01-06,10,12,12,9,100,1000,0,0,0,0",
    ])
    df = parse_eastmoney_klines(p, INSTRUMENT, cutoff="2026-01-05")
    assert df["date"].max() == pd.Timestamp("2026-01-02")

def test_parse_rejects_identity_mismatch():
    p = payload(["2026-01-02,9,10,11,8,90,900,0,0,0,0"], name="别的指数", code="999999")
    with pytest.raises(ValueError, match="identity"):
        parse_eastmoney_klines(p, INSTRUMENT)

def test_parse_rejects_empty_response():
    with pytest.raises(ValueError, match="empty"):
        parse_eastmoney_klines({"data":{"code":"000001","name":"上证指数","klines":[]}}, INSTRUMENT)

def test_fetch_retries_transient_error_and_honors_iso_cutoff(monkeypatch):
    from ma_breakdown.src import data as mod
    class Resp:
        def __init__(self, ok, url="https://example.test"):
            self.ok=ok; self.url=url
        def raise_for_status(self):
            if not self.ok: raise mod.requests.HTTPError("temporary")
        def json(self):
            return payload([
                "2026-07-31,9,10,11,8,90,900,0,0,0,0",
                "2026-08-03,10,12,12,9,100,1000,0,0,0,0",
            ])
    calls=[]
    def fake_get(*args, **kwargs):
        calls.append(1)
        return Resp(ok=len(calls)>1)
    monkeypatch.setattr(mod.requests,"get",fake_get)
    monkeypatch.setattr(mod.time,"sleep",lambda *_: None)
    df,meta=mod.fetch_instrument(INSTRUMENT,"2004-01-01","2026-08-16",cutoff="2026-07-31",retries=2)
    assert len(calls)==2
    assert df.date.max()==pd.Timestamp("2026-07-31")
    assert meta["last_date"]=="2026-07-31"

def tx_text(symbol, rows, key="day"):
    import json
    obj={"data":{symbol:{key:rows,"qt":{symbol:["",symbol]},"prec":"2"}}}
    return "kline_dayqfq="+json.dumps(obj,ensure_ascii=False)

def test_parse_tencent_rows_maps_fields_and_filters_dates():
    from ma_breakdown.src.data import parse_tencent_kline_text
    ins={**INSTRUMENT,"tencent_symbol":"sh000001"}
    text=tx_text("sh000001",[
        ["2004-12-31","1266.50","1266.50","1270.00","1250.00","100"],
        ["2005-01-04","1260.78","1242.77","1260.78","1238.18","200"],
        ["2005-01-05","1241.68","1251.94","1258.83","1235.75","210"],
    ])
    df=parse_tencent_kline_text(text,ins,begin="2005-01-04",end="2005-01-05")
    assert list(df.date.dt.strftime("%Y-%m-%d"))==["2005-01-04","2005-01-05"]
    assert list(df.close)==[1242.77,1251.94]
    assert list(df.columns[:6])==["date","open","close","high","low","amount"]

def test_parse_tencent_rejects_wrong_symbol_or_empty_rows():
    from ma_breakdown.src.data import parse_tencent_kline_text
    ins={**INSTRUMENT,"tencent_symbol":"sh000001"}
    with pytest.raises(ValueError,match="identity"):
        parse_tencent_kline_text(tx_text("sh000016",[["2005-01-04","1","1","1","1","1"]]),ins,"2005-01-01","2005-12-31")
    with pytest.raises(ValueError,match="empty"):
        parse_tencent_kline_text(tx_text("sh000001",[]),ins,"2005-01-01","2005-12-31")

def test_fetch_tencent_stitches_overlapping_windows_dedupes_and_clips(monkeypatch):
    from ma_breakdown.src import data as mod
    ins={**INSTRUMENT,"tencent_symbol":"sh000001"}
    class Resp:
        status_code=200
        def __init__(self,text,url): self.text=text; self.url=url; self.content=text.encode()
        def raise_for_status(self): return None
    calls=[]
    payloads=[
        tx_text("sh000001",[
            ["2004-12-31","9","9","9","9","1"],
            ["2005-01-04","10","10","10","10","1"],
            ["2005-01-05","11","11","11","11","1"],
        ]),
        tx_text("sh000001",[
            ["2005-12-30","11","11","11","11","1"],
            ["2006-01-05","12","12","12","12","1"],
            ["2007-01-02","13","13","13","13","1"],
        ]),
    ]
    def fake_get(url,params,headers,timeout):
        calls.append(params["param"])
        return Resp(payloads[min(len(calls)-1,1)],url)
    monkeypatch.setattr(mod.requests,"get",fake_get)
    monkeypatch.setattr(mod.time,"sleep",lambda *_:None)
    df,meta=mod.fetch_tencent_instrument(ins,"2005-01-04","2006-12-31",timeout=1,retries=1,window_years=1)
    assert list(df.date.dt.strftime("%Y-%m-%d"))==["2005-01-04","2005-01-05","2006-01-05"]
    assert not df.date.duplicated().any()
    assert meta["source"]=="tencent"
    assert meta["symbol"]=="sh000001"
    assert meta["first_date"]=="2005-01-04" and meta["last_date"]=="2006-01-05"

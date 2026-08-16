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

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
import pandas as pd
import requests
import yaml

EM_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
FIELDS1 = "f1,f2,f3,f4,f5,f6"
FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
COLS = ["date","open","close","high","low","volume","amount","amplitude","pct","change","turnover"]

TX_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
TX_HEADERS = {"User-Agent":"Mozilla/5.0", "Referer":"https://gu.qq.com/"}

def parse_tencent_kline_text(text: str, instrument: dict[str, Any], begin: str, end: str) -> pd.DataFrame:
    symbol = instrument.get("tencent_symbol")
    if not symbol:
        raise ValueError(f"missing tencent_symbol for {instrument['name']}")
    if "=" not in text:
        raise ValueError(f"invalid Tencent response for {instrument['name']}")
    try:
        obj = json.loads(text.split("=", 1)[1].strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Tencent JSON for {instrument['name']}") from exc
    data = obj.get("data") if isinstance(obj, dict) else None
    if not isinstance(data, dict):
        raise ValueError(f"empty Tencent response for {instrument['name']}")
    if symbol not in data:
        if data:
            raise ValueError(f"identity mismatch: expected Tencent symbol {symbol}, got {list(data)[:3]}")
        raise ValueError(f"empty Tencent response for {instrument['name']}")
    node = data.get(symbol) or {}
    rows = node.get("day") or node.get("qfqday") or []
    if not rows:
        raise ValueError(f"empty Tencent response for {instrument['name']}")
    clean=[]
    for row in rows:
        if len(row) < 6:
            continue
        clean.append(row[:6])
    if not clean:
        raise ValueError(f"empty Tencent response for {instrument['name']}")
    df=pd.DataFrame(clean, columns=["date","open","close","high","low","amount"])
    df["date"]=pd.to_datetime(df["date"], errors="raise")
    for c in ["open","close","high","low","amount"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df[(df["date"]>=pd.Timestamp(begin))&(df["date"]<=pd.Timestamp(end))].copy()
    df=df.dropna(subset=["date","close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"empty Tencent response for {instrument['name']}")
    if df["date"].duplicated().any():
        raise ValueError(f"duplicate dates in Tencent chunk for {instrument['name']}")
    df["instrument_key"]=instrument["key"]
    df["instrument_name"]=instrument["name"]
    return df

def fetch_tencent_instrument(instrument: dict[str, Any], begin: str, end: str, timeout: int = 20, retries: int = 3, window_years: int = 2) -> tuple[pd.DataFrame, dict[str, Any]]:
    symbol=instrument.get("tencent_symbol")
    if not symbol:
        raise ValueError(f"missing tencent_symbol for {instrument['name']}")
    start=pd.Timestamp(begin); finish=pd.Timestamp(end)
    if finish < start:
        raise ValueError("end before begin")
    frames=[]; urls=[]
    year=start.year
    step=max(1,int(window_years))
    while year <= finish.year:
        chunk_start=max(start,pd.Timestamp(f"{year}-01-01"))
        last_year=min(finish.year, year+step-1)
        chunk_end=min(finish,pd.Timestamp(f"{last_year}-12-31"))
        params={
            "_var":"kline_dayqfq",
            "param":f"{symbol},day,{chunk_start.date()},{chunk_end.date()},640,qfq",
            "r":"0.8205512681390605",
        }
        last_exc=None; response_url=None
        for attempt in range(max(1,retries)):
            try:
                r=requests.get(TX_URL,params=params,headers=TX_HEADERS,timeout=timeout)
                r.raise_for_status(); response_url=r.url
                try:
                    chunk=parse_tencent_kline_text(r.text,instrument,str(chunk_start.date()),str(chunk_end.date()))
                except ValueError as exc:
                    if "empty Tencent response" in str(exc):
                        chunk=pd.DataFrame()
                    else:
                        raise
                if not chunk.empty:
                    frames.append(chunk)
                urls.append(response_url)
                last_exc=None
                break
            except (requests.RequestException, ValueError, KeyError) as exc:
                last_exc=exc
                if attempt+1 < max(1,retries):
                    time.sleep(1.25*(attempt+1))
        if last_exc is not None:
            raise RuntimeError(f"failed Tencent chunk {instrument['name']} {chunk_start.date()}..{chunk_end.date()}: {last_exc}") from last_exc
        year += step
    if not frames:
        raise RuntimeError(f"no Tencent history for {instrument['name']} ({symbol})")
    df=pd.concat(frames,ignore_index=True)
    df=df[(df["date"]>=start)&(df["date"]<=finish)].sort_values("date")
    df=df.drop_duplicates(subset=["date"],keep="last").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"no Tencent history after clipping for {instrument['name']}")
    meta={
        "source":"tencent", "symbol":symbol, "url":TX_URL, "request_count":len(urls),
        "rows":int(len(df)), "first_date":df.date.min().strftime("%Y-%m-%d"),
        "last_date":df.date.max().strftime("%Y-%m-%d"),
    }
    return df,meta

def load_instruments(path: str | Path) -> list[dict[str, Any]]:
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return obj["instruments"]

def parse_eastmoney_klines(payload: dict[str, Any], instrument: dict[str, Any], cutoff: str | None = None) -> pd.DataFrame:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not data or not data.get("klines"):
        raise ValueError(f"empty Eastmoney response for {instrument['name']}")
    expected_code = str(instrument.get("code", ""))
    expected_name = str(instrument.get("name", ""))
    actual_code = str(data.get("code", ""))
    actual_name = str(data.get("name", ""))
    if actual_code and actual_code != expected_code:
        raise ValueError(f"identity mismatch: expected code {expected_code}, got {actual_code}")
    if actual_name and actual_name != expected_name:
        raise ValueError(f"identity mismatch: expected name {expected_name}, got {actual_name}")
    rows = [x.split(",") for x in data["klines"]]
    df = pd.DataFrame(rows, columns=COLS)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df["date"].duplicated().any():
        raise ValueError(f"duplicate dates in {instrument['name']}")
    df = df.sort_values("date").reset_index(drop=True)
    if cutoff:
        df = df[df["date"] <= pd.Timestamp(cutoff)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"empty after cutoff for {instrument['name']}")
    df["instrument_key"] = instrument["key"]
    df["instrument_name"] = instrument["name"]
    return df

def fetch_instrument(instrument: dict[str, Any], begin: str, end: str, cutoff: str | None = None, timeout: int = 30, retries: int = 3) -> tuple[pd.DataFrame, dict[str, Any]]:
    params = {
        "secid": instrument["secid"], "klt":"101", "fqt":"0",
        "beg": begin.replace("-", ""), "end": end.replace("-", ""), "lmt":"1000000",
        "fields1": FIELDS1, "fields2": FIELDS2,
        "ut": "fa5fd1943c7b386f172d6893dbbd1d0c"
    }
    headers = {"User-Agent":"Mozilla/5.0", "Referer":"https://quote.eastmoney.com/"}
    last_exc=None
    for attempt in range(max(1,retries)):
        try:
            r = requests.get(EM_URL, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            df = parse_eastmoney_klines(payload, instrument, cutoff=cutoff or end)
            meta = {
                "source":"eastmoney", "url": r.url, "secid": instrument["secid"],
                "code": str(payload.get("data",{}).get("code","")),
                "name": str(payload.get("data",{}).get("name","")),
                "rows": int(len(df)), "first_date": df.date.min().strftime("%Y-%m-%d"),
                "last_date": df.date.max().strftime("%Y-%m-%d"),
            }
            return df, meta
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_exc=exc
            if attempt+1 >= max(1,retries):
                break
            time.sleep(1.5*(attempt+1))
    raise RuntimeError(f"failed to fetch {instrument['name']} after {retries} attempts: {last_exc}") from last_exc

def save_raw(df: pd.DataFrame, out: str | Path) -> None:
    p = Path(out); p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

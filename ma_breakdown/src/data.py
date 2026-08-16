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

from __future__ import annotations
import pandas as pd

EVENT_COLUMNS = [
    "d1_idx","d2_idx","d3_idx","d1_date","d2_date","d3_date","d1_close","d2_close","d3_close",
    "d1_ma","d2_ma","d3_ma","peak_idx","peak_date","peak_close","trough_idx","trough_date","trough_close",
    "r1_idx","r2_idx","r3_idx","r1_date","r2_date","r3_date","r1_close","r2_close","r3_close"
]

def compute_ma(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"].rolling(window=window, min_periods=window).mean()

def _eligible(df, i, ma_col):
    return i > 0 and pd.notna(df.at[i, ma_col]) and pd.notna(df.at[i-1, ma_col])

def _is_d1(df, i, ma_col):
    return _eligible(df,i,ma_col) and df.at[i-1,"close"] >= df.at[i-1,ma_col] and df.at[i,"close"] < df.at[i,ma_col]

def detect_candidate_stats(df: pd.DataFrame, ma_col: str, sample_start: str, cutoff: str) -> dict:
    rec=detect_candidate_records(df, ma_col, sample_start, cutoff)
    d1=len(rec)
    d2=int(rec["reached_day2"].sum()) if d1 else 0
    true=int(rec["true_breakdown"].sum()) if d1 else 0
    return {
        "day1_candidates":d1,
        "day2_reached":d2,
        "true_breakdowns":true,
        "day1_success_rate": (true/d1 if d1 else None),
        "day2_success_rate": (true/d2 if d2 else None),
    }

def _event_row(x, i, r3_idx, ma_col, peak_lookback):
    peak_slice=x.iloc[i-peak_lookback:i]
    peak_close=peak_slice["close"].max()
    peak_candidates=peak_slice.index[peak_slice["close"].eq(peak_close)]
    peak_idx=int(peak_candidates.max())
    trough_slice=x.iloc[i:r3_idx+1]
    trough_close=trough_slice["close"].min()
    trough_idx=int(trough_slice.index[trough_slice["close"].eq(trough_close)].min())
    r1_idx=r3_idx-2; r2_idx=r3_idx-1
    return {
        "d1_idx":i,"d2_idx":i+1,"d3_idx":i+2,
        "d1_date":x.at[i,"date"],"d2_date":x.at[i+1,"date"],"d3_date":x.at[i+2,"date"],
        "d1_close":float(x.at[i,"close"]),"d2_close":float(x.at[i+1,"close"]),"d3_close":float(x.at[i+2,"close"]),
        "d1_ma":float(x.at[i,ma_col]),"d2_ma":float(x.at[i+1,ma_col]),"d3_ma":float(x.at[i+2,ma_col]),
        "peak_idx":peak_idx,"peak_date":x.at[peak_idx,"date"],"peak_close":float(peak_close),
        "trough_idx":trough_idx,"trough_date":x.at[trough_idx,"date"],"trough_close":float(trough_close),
        "r1_idx":r1_idx,"r2_idx":r2_idx,"r3_idx":r3_idx,
        "r1_date":x.at[r1_idx,"date"],"r2_date":x.at[r2_idx,"date"],"r3_date":x.at[r3_idx,"date"],
        "r1_close":float(x.at[r1_idx,"close"]),"r2_close":float(x.at[r2_idx,"close"]),"r3_close":float(x.at[r3_idx,"close"]),
    }

def detect_events(df: pd.DataFrame, ma_col: str, sample_start: str, cutoff: str, peak_lookback: int=10):
    x=df.sort_values("date").reset_index(drop=True).copy()
    start,end=pd.Timestamp(sample_start),pd.Timestamp(cutoff)
    completed=[]; ongoing=[]
    i=1
    while i < len(x):
        if x.at[i,"date"] > end: break
        if x.at[i,"date"] < start or not _is_d1(x,i,ma_col):
            i += 1; continue
        if i < peak_lookback:
            i += 1; continue
        if i+2 >= len(x) or x.at[i+2,"date"] > end:
            i += 1; continue
        if not (x.at[i+1,"close"] < x.at[i+1,ma_col] and x.at[i+2,"close"] < x.at[i+2,ma_col]):
            i += 1; continue
        run=0; r3_idx=None
        j=i+3
        while j < len(x) and x.at[j,"date"] <= end:
            if pd.notna(x.at[j,ma_col]) and x.at[j,"close"] >= x.at[j,ma_col]:
                run += 1
                if run == 3:
                    r3_idx=j; break
            else:
                run=0
            j += 1
        if r3_idx is None:
            row={
                "d1_idx":i,"d2_idx":i+1,"d3_idx":i+2,
                "d1_date":x.at[i,"date"],"d2_date":x.at[i+1,"date"],"d3_date":x.at[i+2,"date"],
                "d1_close":float(x.at[i,"close"]),"d2_close":float(x.at[i+1,"close"]),"d3_close":float(x.at[i+2,"close"]),
                "d1_ma":float(x.at[i,ma_col]),"d2_ma":float(x.at[i+1,ma_col]),"d3_ma":float(x.at[i+2,ma_col]),
            }
            ongoing.append(row); break
        completed.append(_event_row(x,i,r3_idx,ma_col,peak_lookback))
        i=r3_idx+1
    return pd.DataFrame(completed, columns=EVENT_COLUMNS), pd.DataFrame(ongoing)

def detect_candidate_records(df: pd.DataFrame, ma_col: str, sample_start: str, cutoff: str) -> pd.DataFrame:
    """Return one candidate record per eligible D1, suppressing crosses while a true event is active."""
    x=df.sort_values("date").reset_index(drop=True)
    start,end=pd.Timestamp(sample_start),pd.Timestamp(cutoff)
    rows=[]
    i=1
    while i < len(x):
        d=x.at[i,"date"]
        if d > end:
            break
        if d < start or not _is_d1(x,i,ma_col):
            i += 1
            continue
        reached_day2 = bool(
            i+1 < len(x) and x.at[i+1,"date"] <= end
            and pd.notna(x.at[i+1,ma_col])
            and x.at[i+1,"close"] < x.at[i+1,ma_col]
        )
        true_breakdown = bool(
            reached_day2 and i+2 < len(x) and x.at[i+2,"date"] <= end
            and pd.notna(x.at[i+2,ma_col])
            and x.at[i+2,"close"] < x.at[i+2,ma_col]
        )
        rows.append({
            "d1_idx":i,"d1_date":d,"year":int(d.year),
            "reached_day2":reached_day2,"true_breakdown":true_breakdown
        })
        if not true_breakdown:
            i += 1
            continue
        run=0; r3_idx=None
        j=i+3
        while j < len(x) and x.at[j,"date"] <= end:
            if pd.notna(x.at[j,ma_col]) and x.at[j,"close"] >= x.at[j,ma_col]:
                run += 1
                if run == 3:
                    r3_idx=j
                    break
            else:
                run=0
            j += 1
        if r3_idx is None:
            break
        i=r3_idx+1
    return pd.DataFrame(rows, columns=["d1_idx","d1_date","year","reached_day2","true_breakdown"])

import pandas as pd
from ma_breakdown.src.events import detect_events, detect_candidate_stats

def frame(closes, mas=None, start="2026-01-01"):
    if mas is None: mas=[100.0]*len(closes)
    return pd.DataFrame({"date":pd.bdate_range(start, periods=len(closes)), "close":closes, "ma":mas})

def test_true_breakdown_and_three_day_recovery_with_false_recovery_inside():
    closes=[101,102,103,104,105,106,107,108,110,109, 99,98,97,90,95,101,102,99,92,91,101,102,103]
    df=frame(closes)
    events, ongoing=detect_events(df,"ma","2026-01-01","2026-12-31",peak_lookback=10)
    assert len(events)==1 and ongoing.empty
    e=events.iloc[0]
    assert e.d1_close==99 and e.d2_close==98 and e.d3_close==97
    assert e.peak_close==110 and e.peak_date==df.iloc[8].date
    assert e.trough_close==90 and e.trough_date==df.iloc[13].date
    assert e.r1_close==101 and e.r3_close==103
    assert e.r1_date==df.iloc[20].date and e.r3_date==df.iloc[22].date

def test_false_breakdown_is_not_main_event_but_counts_candidate():
    closes=[101]*10 + [99,98,101,102,103]
    df=frame(closes)
    events, ongoing=detect_events(df,"ma","2026-01-01","2026-12-31",peak_lookback=10)
    stats=detect_candidate_stats(df,"ma","2026-01-01","2026-12-31")
    assert events.empty and ongoing.empty
    assert stats["day1_candidates"]==1
    assert stats["day2_reached"]==1
    assert stats["true_breakdowns"]==0

def test_duplicate_peak_uses_latest_and_duplicate_trough_uses_earliest():
    closes=[101,110,105,110,108,107,106,105,104,103, 99,98,97,90,90,101,102,103]
    df=frame(closes)
    events,_=detect_events(df,"ma","2026-01-01","2026-12-31",peak_lookback=10)
    e=events.iloc[0]
    assert e.peak_date==df.iloc[3].date
    assert e.trough_date==df.iloc[13].date

def test_true_breakdown_without_r3_is_ongoing_not_completed():
    closes=[101]*10+[99,98,97,90,95,101,99]
    df=frame(closes)
    events,ongoing=detect_events(df,"ma","2026-01-01","2026-12-31",peak_lookback=10)
    assert events.empty
    assert len(ongoing)==1
    assert ongoing.iloc[0].d3_close==97

def test_new_event_can_start_after_completed_r3():
    closes=[101]*10+[99,98,97,90,101,102,103,99,98,97,92,101,102,103]
    df=frame(closes)
    events,_=detect_events(df,"ma","2026-01-01","2026-12-31",peak_lookback=10)
    assert len(events)==2

def test_candidate_records_keep_year_and_truth_flag():
    from ma_breakdown.src.events import detect_candidate_records
    closes=[101]*10+[99,98,101,102,99,98,97,90]
    df=frame(closes)
    rec=detect_candidate_records(df,"ma","2026-01-01","2026-12-31")
    assert len(rec)==2
    assert list(rec["reached_day2"])==[True,True]
    assert list(rec["true_breakdown"])==[False,True]

def test_candidate_records_do_not_double_count_inside_active_true_event():
    from ma_breakdown.src.events import detect_candidate_records
    closes=[101]*10+[99,98,97,90,95,101,102,99,92,91,101,102,103]
    df=frame(closes)
    rec=detect_candidate_records(df,"ma","2026-01-01","2026-12-31")
    assert len(rec)==1
    assert bool(rec.iloc[0].true_breakdown)

from __future__ import annotations
EXPECTED={"上证指数","上证50","沪深300","中证1000","创业板指","科创50","中证红利"}

def validate_payload(payload: dict) -> list[str]:
    errors=[]
    names={x.get("name") for x in payload.get("instruments",[])}
    if "中证全指" in names: errors.append("指数池不得包含中证全指")
    if names != EXPECTED: errors.append(f"指数池不匹配: {sorted(names)}")
    meta=payload.get("meta",{})
    if meta.get("cutoff")!="2026-07-31": errors.append("数据截止日必须为2026-07-31")
    if meta.get("event_count") != len(payload.get("events",[])):
        errors.append("event_count 与事件明细不一致")
    for key in ("overview","depth_rows","conditional_panels","duration","period_rows","conduction_rows","charts","data_quality"):
        if key not in payload: errors.append(f"缺少payload字段: {key}")
    return errors

from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .validate import validate_payload

def pct(v, digits=1):
    if v is None: return "—"
    try:
        if v != v: return "—"
        return f"{v*100:+.{digits}f}%"
    except Exception: return "—"

def num(v, digits=1):
    if v is None: return "—"
    try:
        if v != v: return "—"
        return f"{v:.{digits}f}"
    except Exception: return "—"

def render_report(payload: dict, template_path: str|Path, output_path: str|Path) -> None:
    errors=validate_payload(payload)
    if errors: raise ValueError("payload validation failed: "+"; ".join(errors))
    tp=Path(template_path); out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
    env=Environment(loader=FileSystemLoader(str(tp.parent)),autoescape=select_autoescape(["html"]),trim_blocks=True,lstrip_blocks=True)
    env.filters["pct"]=pct;env.filters["num"]=num
    html=env.get_template(tp.name).render(**payload)
    if "{{" in html or "TODO" in html: raise ValueError("unrendered placeholder in HTML")
    out.write_text(html,encoding="utf-8")

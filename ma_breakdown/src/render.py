from __future__ import annotations
import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .release_patch import apply_v02_release_patch
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


def _strip_report_helper_notes(html: str) -> str:
    """Remove implementation/UI helper copy while preserving the underlying scroll behavior."""
    html=html.replace(
        '<p>宽表保留完整研究字段，不为适配屏幕宽度删除指标。请直接横向滚动查看甲1/甲2/甲3/乙/丙与时间维度。</p>',
        ''
    )
    html=re.sub(r'<p class="scroll-note">.*?</p>', '', html, flags=re.S)
    return html


def render_report(payload: dict, template_path: str|Path, output_path: str|Path) -> None:
    errors=validate_payload(payload)
    if errors: raise ValueError("payload validation failed: "+"; ".join(errors))
    tp=Path(template_path); out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
    env=Environment(loader=FileSystemLoader(str(tp.parent)),autoescape=select_autoescape(["html"]),trim_blocks=True,lstrip_blocks=True)
    env.filters["pct"]=pct;env.filters["num"]=num
    html=env.get_template(tp.name).render(**payload)
    html=html.replace("Day1、Day2、Day3连续3个交易日Close均低于各自均线", "Day1、Day2、Day3连续3个交易日收盘低于对应均线（Close均低于各自均线）")
    html=_strip_report_helper_notes(html)
    html=apply_v02_release_patch(html,payload)
    if "{{" in html or "TODO" in html: raise ValueError("unrendered placeholder in HTML")
    out.write_text(html,encoding="utf-8")

from __future__ import annotations
from html import escape
import math
import pandas as pd


def _svg_axes(width=920,height=300,pad=44):
    return width,height,pad


def line_svg(series: dict[str, list[tuple[float,float]]], y_percent=False, width=920, height=300) -> str:
    width,height,pad=_svg_axes(width,height)
    pts=[(x,y) for vals in series.values() for x,y in vals if y is not None and not pd.isna(y)]
    if not pts:
        return f'<svg viewBox="0 0 {width} {height}"><text x="20" y="35" fill="#86868b">无可用数据</text></svg>'
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
    if math.isclose(xmin,xmax): xmax=xmin+1
    if math.isclose(ymin,ymax): ymax=ymin+1
    ym=(ymax-ymin)*.10; ymin-=ym; ymax+=ym
    sx=lambda x: pad+(x-xmin)/(xmax-xmin)*(width-2*pad)
    sy=lambda y: height-pad-(y-ymin)/(ymax-ymin)*(height-2*pad)
    colors=['#007aff','#e74c3c','#27ae60','#e67e22','#5856d6','#8e8e93']
    parts=[f'<svg viewBox="0 0 {width} {height}" role="img">',f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#d2d2d7"/>',f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#d2d2d7"/>']
    for j,(label,vals) in enumerate(series.items()):
        valid=[(x,y) for x,y in vals if y is not None and not pd.isna(y)]
        if not valid: continue
        coords=' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in valid); c=colors[j%len(colors)]
        parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.2" points="{coords}"/>')
        lx=pad+j*145; parts.append(f'<line x1="{lx}" y1="18" x2="{lx+18}" y2="18" stroke="{c}" stroke-width="3"/><text x="{lx+24}" y="22" font-size="12" fill="#6e6e73">{escape(label)}</text>')
    for x in sorted(set(xs)):
        if len(set(xs))>15 and x not in (xmin,xmax): continue
        parts.append(f'<text x="{sx(x):.1f}" y="{height-14}" text-anchor="middle" font-size="10" fill="#86868b">{int(x) if float(x).is_integer() else x:g}</text>')
    for frac in (0,.5,1):
        y=ymin+(ymax-ymin)*frac; label=f'{y*100:.1f}%' if y_percent else f'{y:.1f}'
        parts.append(f'<text x="{pad-7}" y="{sy(y)+4:.1f}" text-anchor="end" font-size="10" fill="#86868b">{label}</text>')
    parts.append('</svg>'); return ''.join(parts)


def event_path_svg(prices: pd.DataFrame, event: pd.Series|dict, width=920, height=330) -> str:
    if event is None or len(prices)==0: return line_svg({})
    e=dict(event); start=max(0,int(e['peak_idx'])-2); p1=e.get('p1_idx'); end=int(e['r3_idx'])
    if p1 is not None and not pd.isna(p1): end=max(end,int(p1))
    end=min(end,len(prices)-1); segment=prices.iloc[start:end+1].copy()
    if segment.empty: return line_svg({})
    base=float(e['d1_close']); step=max(1,len(segment)//240)
    vals=[(int(i),float(prices.at[i,'close']/base*100)) for i in segment.index[::step]]
    width,height,pad=_svg_axes(width,height); xs=[x for x,_ in vals]; ys=[y for _,y in vals]
    xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
    if xmin==xmax: xmax=xmin+1
    if ymin==ymax: ymax=ymin+1
    ym=(ymax-ymin)*.12; ymin-=ym; ymax+=ym
    sx=lambda x: pad+(x-xmin)/(xmax-xmin)*(width-2*pad); sy=lambda y: height-pad-(y-ymin)/(ymax-ymin)*(height-2*pad)
    coords=' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in vals)
    parts=[f'<svg viewBox="0 0 {width} {height}" role="img"><polyline fill="none" stroke="#007aff" stroke-width="2.2" points="{coords}"/>']
    markers=[('Peak','peak_idx','#5856d6'),('D1','d1_idx','#e67e22'),('D2','d2_idx','#e67e22'),('D3','d3_idx','#e74c3c'),('Trough','trough_idx','#c0392b'),('R1','r1_idx','#27ae60'),('R3','r3_idx','#188038'),('P1','p1_idx','#007aff')]
    for label,k,c in markers:
        idx=e.get(k)
        if idx is None or pd.isna(idx): continue
        idx=int(idx)
        if idx<start or idx>end: continue
        y=float(prices.at[idx,'close']/base*100)
        parts.append(f'<circle cx="{sx(idx):.1f}" cy="{sy(y):.1f}" r="4" fill="{c}"/><text x="{sx(idx)+5:.1f}" y="{sy(y)-7:.1f}" font-size="10" fill="{c}">{label}</text>')
    parts.append(f'<text x="{pad}" y="20" font-size="11" fill="#86868b">D1=100 归一化；价格口径=Close</text></svg>')
    return ''.join(parts)

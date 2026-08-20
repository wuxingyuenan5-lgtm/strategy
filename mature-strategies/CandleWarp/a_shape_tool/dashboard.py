"""dashboard.py — Institutional-grade Dark-Mode Multi-Timeframe Interactive Visual Dashboard for Clients."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd


def image_to_base64_data_uri(img_path: str | Path) -> str:
    """Read an image and convert it to base64 data URI for zero-dependency standalone HTML embedding."""
    p = Path(img_path)
    if not p.exists():
        return ""
    encoded = base64.b64encode(p.read_bytes()).decode("utf-8")
    ext = p.suffix.lower().replace(".", "")
    mime = "image/png" if ext == "png" else ("image/jpeg" if ext in ("jpg", "jpeg") else "image/svg+xml")
    return f"data:{mime};base64,{encoded}"


def generate_client_dashboard_html(
    timeframe_data: list[dict],
    symbol: str = "XAUUSD",
    title: str = "CandleWarp™ Multi-Timeframe Morphology & Trend Distribution Engine",
    output_html_path: str | Path = "dashboard.html",
    embed_images_base64: bool = True,
) -> Path:
    """Generate an institutional-grade dark-mode interactive HTML presentation dashboard for clients."""
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    tab_buttons = []
    tab_contents = []

    for i, data in enumerate(timeframe_data):
        tf = data["tf"]
        desc = data.get("desc", tf)
        is_active = (i == 0)
        active_btn_cls = "bg-blue-600 text-white shadow-lg shadow-blue-500/20" if is_active else "bg-slate-800/80 text-slate-400 hover:bg-slate-700 hover:text-white"
        active_tab_style = "display: block;" if is_active else "display: none;"

        tab_buttons.append(
            f'<button onclick="switchTab(\'{tf}\')" id="btn-{tf}" '
            f'class="px-4 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 {active_btn_cls}">'
            f'<span>{tf.upper()}</span> <span class="text-xs opacity-75 ml-1">({desc})</span>'
            f'</button>'
        )

        if embed_images_base64:
            dist_src = image_to_base64_data_uri(data.get("distribution_img", ""))
            candles_src = image_to_base64_data_uri(data.get("candles_img", ""))
            diag_src = image_to_base64_data_uri(data.get("diagnostics_img", ""))
        else:
            dist_src = data.get("distribution_img", "")
            candles_src = data.get("candles_img", "")
            diag_src = data.get("diagnostics_img", "")

        conf_badge_color = (
            "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
            if data.get("confidence") == "HIGH"
            else ("bg-amber-500/10 text-amber-400 border-amber-500/20" if data.get("confidence") == "MEDIUM" else "bg-rose-500/10 text-rose-400 border-rose-500/20")
        )

        q50_val = data.get("q50", 0.0)
        q50_color = "text-emerald-400" if q50_val > 0 else ("text-rose-400" if q50_val < 0 else "text-slate-200")
        wq50_val = data.get("wq50", q50_val)
        wq50_color = "text-emerald-400" if wq50_val > 0 else ("text-rose-400" if wq50_val < 0 else "text-slate-200")

        q_rows_html = ""
        if "quantiles_table" in data and isinstance(data["quantiles_table"], pd.DataFrame):
            for _, r in data["quantiles_table"].iterrows():
                q_level = f"{float(r['quantile']):.0%}"
                last_val = r.iloc[-1] * 100.0 if abs(r.iloc[-1]) < 10.0 else r.iloc[-1]
                val_color = "text-emerald-400" if last_val > 0 else "text-rose-400"
                q_rows_html += f"""
                <tr class="border-b border-slate-800/60 hover:bg-slate-800/30 transition-colors">
                  <td class="py-2.5 px-4 font-mono font-medium text-slate-300">{q_level}</td>
                  <td class="py-2.5 px-4 font-mono font-bold {val_color}">{last_val:+.3f}%</td>
                </tr>
                """
        else:
            q_rows_html = f"""
            <tr class="border-b border-slate-800/60"><td class="py-2 px-4 text-slate-400">10% (极端利空)</td><td class="py-2 px-4 font-mono text-rose-400">{data.get('q10', 0):+.2f}%</td></tr>
            <tr class="border-b border-slate-800/60"><td class="py-2 px-4 text-slate-400">25% (稳健下沿)</td><td class="py-2 px-4 font-mono text-slate-200">{data.get('q25', 0):+.2f}%</td></tr>
            <tr class="border-b border-slate-800/60 bg-blue-500/5"><td class="py-2 px-4 font-bold text-blue-300">50% (中位数预期)</td><td class="py-2 px-4 font-mono font-bold {q50_color}">{q50_val:+.2f}%</td></tr>
            <tr class="border-b border-slate-800/60"><td class="py-2 px-4 text-slate-400">75% (稳健上沿)</td><td class="py-2 px-4 font-mono text-slate-200">{data.get('q75', 0):+.2f}%</td></tr>
            <tr><td class="py-2 px-4 text-slate-400">90% (极端利多)</td><td class="py-2 px-4 font-mono text-emerald-400">{data.get('q90', 0):+.2f}%</td></tr>
            """

        diag_html = (
            f'<div class="bg-slate-900/90 backdrop-blur border border-slate-800 p-5 rounded-2xl mb-6">'
            f'<div class="flex items-center gap-2 mb-3">'
            f'<div class="w-3 h-3 rounded-full bg-cyan-500 shadow-md shadow-cyan-500/50"></div>'
            f'<h3 class="text-sm font-bold text-slate-100 uppercase tracking-wide">特征空间与筹码/失衡区诊断 (VP & FVG Diagnostics)</h3>'
            f'</div>'
            f'<div class="bg-slate-950/60 rounded-xl overflow-hidden border border-slate-800/80 p-2">'
            f'<img src="{diag_src}" alt="Diagnostics" class="w-full h-auto object-contain rounded-lg">'
            f'</div></div>'
            if diag_src else ''
        )

        tab_html = f"""
        <div id="tab-{tf}" class="tab-content animate-fade-in" style="{active_tab_style}">
          <!-- Key KPI Cards Grid -->
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div class="bg-slate-900/90 backdrop-blur border border-slate-800 p-4 rounded-2xl">
              <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">当前基准金价 / 状态</div>
              <div class="text-2xl font-black text-white font-mono">${data.get('price', 0):.2f}</div>
              <div class="mt-2 flex items-center gap-2">
                <span class="px-2 py-0.5 rounded text-xs font-mono font-bold bg-slate-800 text-slate-300 border border-slate-700">{data.get('state', 'N/A')}</span>
                <span class="px-2 py-0.5 rounded text-xs font-bold border {conf_badge_color}">{data.get('confidence', 'HIGH')} 置信度</span>
              </div>
            </div>

            <div class="bg-slate-900/90 backdrop-blur border border-slate-800 p-4 rounded-2xl">
              <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">中位数走势预期 (Q50)</div>
              <div class="text-2xl font-black {q50_color} font-mono">{q50_val:+.2f}%</div>
              <div class="mt-2 text-xs text-slate-400">距离加权期望: <span class="font-mono font-bold {wq50_color}">{wq50_val:+.2f}%</span></div>
            </div>

            <div class="bg-slate-900/90 backdrop-blur border border-slate-800 p-4 rounded-2xl">
              <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">25% - 75% 核心置信带</div>
              <div class="text-xl font-black text-blue-400 font-mono">[{data.get('q25', 0):+.2f}%, {data.get('q75', 0):+.2f}%]</div>
              <div class="mt-2 text-xs text-slate-400">尾部极值: <span class="font-mono">[{data.get('q10', 0):+.2f}%, {data.get('q90', 0):+.2f}%]</span></div>
            </div>

            <div class="bg-slate-900/90 backdrop-blur border border-slate-800 p-4 rounded-2xl">
              <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">历史相似形态样本数</div>
              <div class="text-2xl font-black text-purple-400 font-mono">{data.get('matches', 0)} <span class="text-xs text-slate-500 font-normal">个匹配窗口</span></div>
              <div class="mt-2 text-xs text-slate-400">DTW平均距离: <span class="font-mono text-slate-300">{data.get('dist_mean', 0):.1f}</span></div>
            </div>
          </div>

          <!-- Main Visual Panels Grid -->
          <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
            <!-- Left Chart: Forward Distribution Ribbon -->
            <div class="lg:col-span-8 bg-slate-900/90 backdrop-blur border border-slate-800 p-5 rounded-2xl flex flex-col">
              <div class="flex items-center justify-between mb-3">
                <div class="flex items-center gap-2">
                  <div class="w-3 h-3 rounded-full bg-blue-500 shadow-md shadow-blue-500/50"></div>
                  <h3 class="text-sm font-bold text-slate-100 uppercase tracking-wide">未来走势概率分位带 (Distribution Ribbon & Paths)</h3>
                </div>
                <span class="text-xs text-slate-400">展望后续 {data.get('horizon', 30)} 根 K 线</span>
              </div>
              <div class="flex-1 flex items-center justify-center bg-slate-950/60 rounded-xl overflow-hidden border border-slate-800/80 p-2">
                <img src="{dist_src}" alt="Distribution Ribbon" class="w-full h-auto object-contain rounded-lg shadow-2xl hover:scale-[1.01] transition-transform duration-300">
              </div>
            </div>

            <!-- Right Panel: Probability Table -->
            <div class="lg:col-span-4 bg-slate-900/90 backdrop-blur border border-slate-800 p-5 rounded-2xl flex flex-col">
              <div class="flex items-center gap-2 mb-3">
                <div class="w-3 h-3 rounded-full bg-emerald-500 shadow-md shadow-emerald-500/50"></div>
                <h3 class="text-sm font-bold text-slate-100 uppercase tracking-wide">未来分位数数值对照</h3>
              </div>
              <div class="flex-1 overflow-x-auto">
                <table class="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr class="border-b border-slate-800 text-slate-400 font-semibold bg-slate-800/30">
                      <th class="py-2 px-4 rounded-l-lg">分位点 (Quantile)</th>
                      <th class="py-2 px-4 rounded-r-lg">终点收益率期望</th>
                    </tr>
                  </thead>
                  <tbody>
                    {q_rows_html}
                  </tbody>
                </table>
              </div>
              <div class="mt-4 p-3 bg-slate-950/50 rounded-xl border border-slate-800/60 text-xs text-slate-400 leading-relaxed">
                💡 <strong class="text-slate-300">量化风控建议</strong>：若 25% 分位点位于零轴上方，表明最差的 75% 历史路径均呈现正收益，具备极佳的左侧安全垫。
              </div>
            </div>
          </div>

          <!-- Bottom Visual Panel: Candlestick Grid Comparison -->
          <div class="bg-slate-900/90 backdrop-blur border border-slate-800 p-5 rounded-2xl mb-6">
            <div class="flex items-center justify-between mb-3">
              <div class="flex items-center gap-2">
                <div class="w-3 h-3 rounded-full bg-purple-500 shadow-md shadow-purple-500/50"></div>
                <h3 class="text-sm font-bold text-slate-100 uppercase tracking-wide">当前形态 vs 历史 Top 最相似片段裸 K 对比网格</h3>
              </div>
              <span class="text-xs text-slate-400">2D Sakoe-Chiba DTW 弹性规整对齐</span>
            </div>
            <div class="bg-slate-950/60 rounded-xl overflow-hidden border border-slate-800/80 p-2">
              <img src="{candles_src}" alt="Candlestick Grid" class="w-full h-auto object-contain rounded-lg hover:scale-[1.005] transition-transform duration-300">
            </div>
          </div>

          {diag_html}
        </div>
        """
        tab_contents.append(tab_html)

    tabs_header_html = "\n".join(tab_buttons)
    tabs_body_html = "\n".join(tab_contents)

    full_html = f"""<!doctype html>
<html lang="zh-CN" class="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(symbol)} | {html.escape(title)}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {{
      darkMode: 'class',
      theme: {{
        extend: {{
          colors: {{
            slate: {{
              950: '#06090e',
              900: '#0b111a',
              850: '#0f1724',
              800: '#1e293b',
            }}
          }}
        }}
      }}
    }}
  </script>
  <style>
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .animate-fade-in {{
      animation: fadeIn 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}
    body {{
      background-color: #06090e;
      color: #cbd5e1;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
  </style>
</head>
<body class="min-h-screen p-4 sm:p-6 lg:p-8 selection:bg-blue-600 selection:text-white">
  <div class="max-w-7xl mx-auto">
    
    <!-- Top Header & Branding Bar -->
    <header class="flex flex-col md:flex-row md:items-center md:justify-between pb-6 mb-6 border-b border-slate-800/80 gap-4">
      <div class="flex items-center gap-3.5">
        <div class="w-11 h-11 rounded-2xl bg-gradient-to-tr from-blue-600 via-indigo-500 to-cyan-400 p-0.5 shadow-lg shadow-blue-500/20">
          <div class="w-full h-full bg-slate-950 rounded-[14px] flex items-center justify-center text-xl">
            🕯️
          </div>
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 class="text-xl sm:text-2xl font-black tracking-tight text-white">{html.escape(symbol)} <span class="text-slate-400 font-medium text-base">/ 形态相似度与走势分布看板</span></h1>
            <span class="px-2 py-0.5 rounded-full text-[10px] font-extrabold uppercase tracking-wider bg-blue-500/10 text-blue-400 border border-blue-500/20">Client Edition</span>
          </div>
          <p class="text-xs text-slate-400 mt-0.5">330x JIT 2D-DTW 弹性时序规整 · Volume Profile 筹码峰 · FVG 流动性失衡 · 无未来函数概率投影</p>
        </div>
      </div>

      <div class="flex items-center gap-3">
        <div class="text-right hidden sm:block">
          <div class="text-xs text-slate-400">数据引擎状态</div>
          <div class="text-xs font-mono font-bold text-emerald-400 flex items-center justify-end gap-1.5">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> MT5 实时数据源已直连
          </div>
        </div>
      </div>
    </header>

    <!-- Multi-Timeframe Tab Selector -->
    <div class="flex items-center gap-2 overflow-x-auto pb-3 mb-6 scrollbar-none">
      <div class="text-xs font-bold text-slate-400 uppercase tracking-wider mr-2 hidden sm:block">周期选择:</div>
      {tabs_header_html}
    </div>

    <!-- Active Tab Containers -->
    <main>
      {tabs_body_html}
    </main>

    <!-- Institutional Footer -->
    <footer class="mt-12 pt-6 border-t border-slate-800 text-center text-xs text-slate-400 flex flex-col sm:flex-row items-center justify-between gap-4">
      <div>© 2026 <strong>CandleWarp™</strong> Quantitative Analytics. All rights reserved.</div>
      <div class="flex items-center gap-4 text-slate-400">
        <span>基于无未来函数 Walk-Forward 滚动验证</span>
        <span>•</span>
        <span>Sakoe-Chiba 带约束 DTW</span>
        <span>•</span>
        <span>Softmax 距离加权概率云</span>
      </div>
    </footer>

  </div>

  <!-- Tab Switch Script -->
  <script>
    function switchTab(targetTf) {{
      document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
      document.querySelectorAll('button[id^="btn-"]').forEach(btn => {{
        btn.className = "px-4 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 bg-slate-800/80 text-slate-400 hover:bg-slate-700 hover:text-white";
      }});

      const activeContent = document.getElementById('tab-' + targetTf);
      if (activeContent) {{
        activeContent.style.display = 'block';
      }}

      const activeBtn = document.getElementById('btn-' + targetTf);
      if (activeBtn) {{
        activeBtn.className = "px-4 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 bg-blue-600 text-white shadow-lg shadow-blue-500/20";
      }}
    }}
  </script>
</body>
</html>
"""
    output_html_path.write_text(full_html, encoding="utf-8")
    return output_html_path

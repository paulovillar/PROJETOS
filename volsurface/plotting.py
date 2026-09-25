"""Visualização Plotly: superfície 3D por ativo, smiles por vencimento e dashboard."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .surface import VolSurface

# Rampa sequencial azul (magnitude de vol): claro = vol baixa, escuro = vol alta.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK, MUTED, GRID = "#1f1f1c", "#6b6a64", "#e4e2dc"
SYNTH_TAG = "DADOS SINTÉTICOS — apenas demonstração"


def _colorscale() -> list[list]:
    return [[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)]


def surface_traces(
    surf: VolSurface, visible: bool = True, colorbar_x: float = 1.02, showlegend: bool = True
) -> list[go.BaseTraceType]:
    """Superfície interpolada + pontos de mercado (IV OTM) em 3D."""
    x = np.exp(surf.k_grid) * 100
    z = surf.vol_grid * 100
    surface = go.Surface(
        x=x,
        y=surf.du_grid,
        z=z,
        colorscale=_colorscale(),
        colorbar=dict(title="IV (%)", thickness=12, len=0.6, x=colorbar_x),
        showlegend=False,
        opacity=0.95,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=False, width=1)),
        hovertemplate="K/F %{x:.1f}%<br>%{y:.0f} DU<br>IV %{z:.2f}%<extra>" + surf.ticker + "</extra>",
        name=f"{surf.ticker} superfície",
        visible=visible,
    )
    pts = pd.concat([s.points.assign(du=s.du, expiry=str(s.expiry)) for s in surf.fitted])
    k_lo, k_hi = surf.k_grid[0], surf.k_grid[-1]
    pts = pts[pts["k"].between(k_lo, k_hi)]
    market = go.Scatter3d(
        x=np.exp(pts["k"]) * 100,
        y=pts["du"],
        z=pts["iv"] * 100,
        mode="markers",
        marker=dict(size=2.5, color=INK, opacity=0.8),
        customdata=np.c_[pts["option_ticker"], pts["strike"], pts["expiry"]],
        hovertemplate="%{customdata[0]}<br>K %{customdata[1]:.2f} · venc %{customdata[2]}<br>IV mercado %{z:.2f}%<extra></extra>",
        name=f"{surf.ticker} mercado",
        visible=visible,
        showlegend=showlegend,
    )
    return [surface, market]


def _scene() -> dict:
    axis = dict(gridcolor=GRID, backgroundcolor="rgba(0,0,0,0)", color=MUTED)
    return dict(
        xaxis=dict(title="Moneyness K/F (%)", **axis),
        yaxis=dict(title="Prazo (DU)", **axis),
        zaxis=dict(title="Vol implícita (%)", **axis),
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        aspectmode="manual",
        aspectratio=dict(x=1.2, y=1.2, z=0.7),
    )


def asset_figure(surf: VolSurface, synthetic: bool) -> go.Figure:
    """Figura por ativo: superfície 3D (esq.) + smiles por vencimento (dir.)."""
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("Superfície (SVI + interpolação em variância total)", "Smiles por vencimento"),
    )
    for tr in surface_traces(surf, colorbar_x=0.57, showlegend=False):
        fig.add_trace(tr, row=1, col=1)

    fitted = surf.fitted
    idx = np.linspace(0, len(ORDINAL_BLUE) - 1, len(fitted)).round().astype(int)
    for s, ci in zip(fitted, idx):
        color = ORDINAL_BLUE[ci]
        kk = np.linspace(surf.k_grid[0], surf.k_grid[-1], 80)
        fig.add_trace(
            go.Scatter(
                x=np.exp(kk) * 100,
                y=np.sqrt(s.svi.total_variance(kk) / s.T) * 100,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{s.expiry:%d/%m/%y} ({s.du} DU)",
                legendgroup=str(s.expiry),
                hovertemplate="K/F %{x:.1f}%<br>IV SVI %{y:.2f}%<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=2,
        )
        m = s.points["k"].between(surf.k_grid[0], surf.k_grid[-1])
        fig.add_trace(
            go.Scatter(
                x=np.exp(s.points.loc[m, "k"]) * 100,
                y=s.points.loc[m, "iv"] * 100,
                mode="markers",
                marker=dict(color=color, size=8, line=dict(color="white", width=2)),
                legendgroup=str(s.expiry),
                showlegend=False,
                hovertemplate="K/F %{x:.1f}%<br>IV mercado %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )
    title = f"{surf.ticker} — superfície de volatilidade implícita · {surf.ref_date:%d/%m/%Y}"
    if synthetic:
        title += f"<br><sup style='color:#b3261e'>{SYNTH_TAG}</sup>"
    fig.update_layout(
        title=dict(text=title, x=0.01),
        scene=_scene(),
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", color=INK, size=12),
        legend=dict(title="Vencimento", font=dict(size=11)),
        height=680,
        margin=dict(l=10, r=10, t=90, b=10),
    )
    fig.update_xaxes(title="Moneyness K/F (%)", gridcolor=GRID, row=1, col=2)
    fig.update_yaxes(title="Vol implícita (%)", gridcolor=GRID, row=1, col=2)
    return fig


def dashboard_figure(surfaces: list[VolSurface], synthetic: bool) -> go.Figure:
    """Uma figura 3D com menu para alternar entre os ativos."""
    fig = go.Figure()
    for i, surf in enumerate(surfaces):
        for tr in surface_traces(surf, visible=(i == 0)):
            fig.add_trace(tr)
    n = len(surfaces)
    buttons = []
    for i, surf in enumerate(surfaces):
        vis = [False] * (2 * n)
        vis[2 * i] = vis[2 * i + 1] = True
        buttons.append(dict(label=surf.ticker, method="update", args=[{"visible": vis}, {"title.text": _dash_title(surf, synthetic)}]))
    fig.update_layout(
        title=dict(text=_dash_title(surfaces[0], synthetic), x=0.01),
        updatemenus=[dict(buttons=buttons, direction="down", x=0.01, y=1.0, xanchor="left", yanchor="top", showactive=True)],
        scene=_scene(),
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", color=INK, size=12),
        height=720,
        margin=dict(l=10, r=10, t=80, b=10),
    )
    return fig


def _dash_title(surf: VolSurface, synthetic: bool) -> str:
    t = f"{surf.ticker} · superfície de vol implícita · {surf.ref_date:%d/%m/%Y}"
    return t + (f"<br><sup>{SYNTH_TAG}</sup>" if synthetic else "")


def write_dashboard(
    surfaces: list[VolSurface], summary: pd.DataFrame, out: Path, synthetic: bool, source: str, plotlyjs: str | bool = "cdn"
) -> Path:
    """HTML único: seletor de ativo 3D + tabela de métricas + links para páginas por ativo.

    ``plotlyjs``: ``"cdn"`` (leve, requer internet) ou ``True`` (embute plotly.js, funciona offline).
    """
    fig_html = dashboard_figure(surfaces, synthetic).to_html(include_plotlyjs=plotlyjs, full_html=False)
    fmt = summary.copy()
    for c in ("atm_1m", "atm_3m", "atm_6m", "skew_90_110_3m", "term_3m_1m"):
        fmt[c] = (fmt[c] * 100).map("{:.2f}".format)
    fmt["spot"] = fmt["spot"].map(lambda v: "—" if pd.isna(v) else f"{v:,.2f}")
    fmt["rmse_vol_bp"] = fmt["rmse_vol_bp"].map("{:.1f}".format)
    fmt["ticker"] = fmt["ticker"].map(lambda t: f'<a href="surfaces/{t}.html">{html.escape(t)}</a>')
    fmt.columns = ["Ativo", "Spot", "ATM 1M", "ATM 3M", "ATM 6M", "Skew 90–110 3M", "3M − 1M", "Venc.", "Pontos", "RMSE fit (bp)"]
    table = fmt.to_html(index=False, escape=False, border=0, classes="tbl")
    banner = f'<div class="warn">{SYNTH_TAG}. Rode com <code>--source cotahist</code> para dados da B3.</div>' if synthetic else ""
    page = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Superfícies de Vol IBOV</title>
<style>
:root{{--bg:#fbfaf7;--ink:#1f1f1c;--muted:#6b6a64;--line:#e4e2dc;--warn:#b3261e}}
body{{background:var(--bg);color:var(--ink);font:14px/1.45 Inter,system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1200px;margin-inline:auto}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:var(--muted);margin-bottom:16px}}
.warn{{border:1px solid var(--warn);color:var(--warn);padding:8px 12px;border-radius:6px;margin-bottom:16px}}
.wrap{{overflow-x:auto}} .tbl{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
.tbl th,.tbl td{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child{{text-align:left}} .tbl th{{color:var(--muted);font-weight:600}}
a{{color:#256abf}}
</style></head><body>
<h1>Superfícies de volatilidade implícita — Ibovespa e 20 maiores pesos</h1>
<div class="sub">Data-base {surfaces[0].ref_date:%d/%m/%Y} · fonte: {html.escape(source)} · vol em % a.a. (DU/252) · moneyness forward K/F</div>
{banner}{fig_html}
<h2 style="font-size:16px">Resumo</h2>
<div class="wrap">{table}</div>
<p class="sub">Skew 90–110: IV(K/F=90%) − IV(K/F=110%) no prazo de 63 DU. Clique no ativo para superfície + smiles por vencimento.</p>
</body></html>"""
    out.write_text(page, encoding="utf-8")
    return out

"""Optimiser page Plotly figures."""

from __future__ import annotations

import plotly.graph_objects as go

from components import CHANNEL_COLORS, apply_dark_theme

def _allocation_donut(
    channels: list[str], alloc: dict[str, float], *, title: str
) -> go.Figure:
    values = [max(0.0, float(alloc.get(c, 0.0))) for c in channels]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=channels,
                values=values,
                hole=0.55,
                marker=dict(colors=[CHANNEL_COLORS[c] for c in channels]),
                textinfo="label+percent",
                textposition="outside",
                hovertemplate="%{label}<br>%{percent}<br>$%{value:,.0f} / week<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5, xanchor="center"),
        showlegend=False,
        margin=dict(t=44, b=16, l=16, r=16),
    )
    return apply_dark_theme(fig, height=300)


def _weights_from_weekly_alloc(
    channels: list[str], alloc: dict[str, float]
) -> dict[str, float]:
    total = sum(alloc.get(c, 0.0) for c in channels) or 1.0
    return {c: float(alloc[c]) / total * 100.0 for c in channels}


def _default_weights(result: ModelResult) -> dict[str, float]:
    alloc = _current_weekly_alloc(result)
    return _weights_from_weekly_alloc(result.channels, alloc)


def _fmt_weight(weight: float) -> str:
    return f"{weight:.1f}% weight"

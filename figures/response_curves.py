"""Response curve Plotly figures."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from components import (
    CHANNEL_COLORS,
    CHART_FONT_COLOR,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    apply_dark_theme,
    with_alpha,
)
from model.mmm import ModelResult, observed_spend_support, response_curve

def response_curve_figure(result: ModelResult, channel: str) -> go.Figure:
    grid, contrib, current, sat_90, hdi_low, hdi_high = response_curve(result, channel)
    support_p5, support_p95 = observed_spend_support(result, channel)
    color = CHANNEL_COLORS[channel]

    fig = go.Figure()

    if len(grid):
        grid_min = float(grid.min())
        grid_max = float(grid.max())
        if support_p5 > grid_min:
            fig.add_vrect(
                x0=grid_min,
                x1=min(support_p5, grid_max),
                fillcolor=with_alpha(NEGATIVE_COLOR, 0.07),
                line_width=0,
                annotation_text="Below observed support",
                annotation_position="top left",
                annotation_font=dict(color=NEGATIVE_COLOR, size=10),
            )
        if support_p95 < grid_max:
            fig.add_vrect(
                x0=max(support_p95, grid_min),
                x1=grid_max,
                fillcolor=with_alpha(NEGATIVE_COLOR, 0.07),
                line_width=0,
                annotation_text="Above observed support",
                annotation_position="top right",
                annotation_font=dict(color=NEGATIVE_COLOR, size=10),
            )

    # 94% HDI band first (drawn underneath the mean line).
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([grid, grid[::-1]]),
            y=np.concatenate([hdi_high, hdi_low[::-1]]),
            fill="toself",
            fillcolor=with_alpha(color, 0.18),
            line=dict(color="rgba(0,0,0,0)"),
            name="94% HDI",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=grid,
            y=contrib,
            mode="lines",
            name="Posterior mean",
            line=dict(color=color, width=2.5),
            hovertemplate="Weekly spend: $%{x:,.0f}<br>Weekly revenue: $%{y:,.0f}<extra></extra>",
        )
    )

    max_y = float(max(contrib.max(), hdi_high.max())) if len(contrib) else 1.0

    fig.add_vrect(
        x0=support_p5,
        x1=support_p95,
        fillcolor=with_alpha(POSITIVE_COLOR, 0.06),
        line_width=0,
        layer="below",
    )
    fig.add_vline(
        x=support_p5,
        line=dict(color=POSITIVE_COLOR, width=1, dash="dash"),
        annotation_text="P5 observed",
        annotation_position="bottom left",
        annotation_font_color=POSITIVE_COLOR,
    )
    fig.add_vline(
        x=support_p95,
        line=dict(color=POSITIVE_COLOR, width=1, dash="dash"),
        annotation_text="P95 observed",
        annotation_position="bottom right",
        annotation_font_color=POSITIVE_COLOR,
    )

    fig.add_vline(
        x=current,
        line=dict(color=CHART_FONT_COLOR, width=1, dash="dot"),
        annotation_text=f"Current ${current/1e3:,.0f}K",
        annotation_position="top",
        annotation_font_color=CHART_FONT_COLOR,
    )

    if sat_90 < grid.max():
        fig.add_vline(
            x=sat_90,
            line=dict(color=NEGATIVE_COLOR, width=1, dash="dash"),
            annotation_text=f"90% saturation ${sat_90/1e3:,.0f}K",
            annotation_position="top right",
            annotation_font_color=NEGATIVE_COLOR,
        )
        fig.add_vrect(
            x0=sat_90,
            x1=grid.max(),
            fillcolor=with_alpha(NEGATIVE_COLOR, 0.06),
            line_width=0,
            annotation_text="Diminishing returns",
            annotation_position="top left",
            annotation_font=dict(color=NEGATIVE_COLOR, size=11),
        )

    fig.update_layout(
        xaxis_title="Weekly spend ($)",
        yaxis_title="Incremental weekly revenue ($)",
        xaxis_tickprefix="$",
        xaxis_tickformat=".2s",
        yaxis_tickprefix="$",
        yaxis_tickformat=".2s",
        showlegend=True,
    )
    fig.update_yaxes(range=[0, max_y * 1.15])
    return apply_dark_theme(fig, height=420)

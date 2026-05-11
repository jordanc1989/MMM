"""Response Curves page: per-channel saturation with diminishing-returns marker."""

from __future__ import annotations

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc

from components import CHANNEL_COLORS, apply_dark_theme, page_header, section
from components.ids import MODEL_REFRESH_STORE
from model.mmm import ModelResult, observed_spend_support, response_curve


CHANNEL_SELECT_ID = "response-curve-channel"
CHANNEL_GRAPH_ID = "response-curve-graph"
CHANNEL_STATS_ID = "response-curve-stats"


# ---------- charts ---------------------------------------------------------


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
                fillcolor="rgba(239, 68, 68, 0.06)",
                line_width=0,
                annotation_text="Below observed support",
                annotation_position="top left",
                annotation_font=dict(color="#f97316", size=10),
            )
        if support_p95 < grid_max:
            fig.add_vrect(
                x0=max(support_p95, grid_min),
                x1=grid_max,
                fillcolor="rgba(239, 68, 68, 0.06)",
                line_width=0,
                annotation_text="Above observed support",
                annotation_position="top right",
                annotation_font=dict(color="#f97316", size=10),
            )

    # 94% HDI band first (drawn underneath the mean line).
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([grid, grid[::-1]]),
            y=np.concatenate([hdi_high, hdi_low[::-1]]),
            fill="toself",
            fillcolor=_hex_with_alpha(color, 0.18),
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
        fillcolor="rgba(45, 212, 191, 0.06)",
        line_width=0,
        layer="below",
    )
    fig.add_vline(
        x=support_p5,
        line=dict(color="#2dd4bf", width=1, dash="dash"),
        annotation_text="P5 observed",
        annotation_position="bottom left",
        annotation_font_color="#2dd4bf",
    )
    fig.add_vline(
        x=support_p95,
        line=dict(color="#2dd4bf", width=1, dash="dash"),
        annotation_text="P95 observed",
        annotation_position="bottom right",
        annotation_font_color="#2dd4bf",
    )

    fig.add_vline(
        x=current,
        line=dict(color="#d4d6d9", width=1, dash="dot"),
        annotation_text=f"Current ${current/1e3:,.0f}K",
        annotation_position="top",
        annotation_font_color="#d4d6d9",
    )

    if sat_90 < grid.max():
        fig.add_vline(
            x=sat_90,
            line=dict(color="#f59e0b", width=1, dash="dash"),
            annotation_text=f"90% saturation ${sat_90/1e3:,.0f}K",
            annotation_position="top right",
            annotation_font_color="#f59e0b",
        )
        fig.add_vrect(
            x0=sat_90,
            x1=grid.max(),
            fillcolor="rgba(245, 158, 11, 0.06)",
            line_width=0,
            annotation_text="Diminishing returns",
            annotation_position="top left",
            annotation_font=dict(color="#f59e0b", size=11),
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


def _hex_with_alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------- stats strip ----------------------------------------------------


def _fmt_currency(v: float) -> str:
    if abs(v) >= 1e6:
        return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:.1f}K"
    return f"${v:,.0f}"


def response_stats(result: ModelResult, channel: str) -> dmc.SimpleGrid:
    grid, contrib, current, sat_90, _lo, _hi = response_curve(result, channel)
    support_p5, support_p95 = observed_spend_support(result, channel)
    # Interpolate mean posterior curve at the current average weekly spend.
    cur_weekly_rev = float(np.interp(current, grid, contrib)) if len(grid) else 0.0
    if len(grid) >= 3:
        idx = int(np.searchsorted(grid, current))
        idx = int(np.clip(idx, 1, len(grid) - 2))
        marginal_roas = float(
            (contrib[idx + 1] - contrib[idx - 1])
            / (grid[idx + 1] - grid[idx - 1])
        )
    else:
        marginal_roas = 0.0
    headroom = max(0.0, (sat_90 - current) / current) if current > 0 else 0.0

    stats = [
        ("Avg weekly spend", _fmt_currency(current)),
        ("Current weekly contribution", _fmt_currency(cur_weekly_rev)),
        ("Marginal ROAS at avg spend", f"{marginal_roas:.2f}x"),
        ("Observed spend range", f"{_fmt_currency(support_p5)}–{_fmt_currency(support_p95)}"),
        ("Headroom to 90% saturation", f"{headroom*100:,.0f}%" if current > 0 else "—"),
    ]
    return dmc.SimpleGrid(
        cols={"base": 2, "md": 5},
        spacing="md",
        children=[
            dmc.Stack(
                gap=2,
                children=[
                    dmc.Text(label, size="xs", c="dimmed", tt="uppercase", fw=600),
                    dmc.Text(value, size="lg", fw=600, className="mmm-numeric"),
                ],
            )
            for label, value in stats
        ],
    )


# ---------- layout + callbacks --------------------------------------------


def build_response_curves(result: ModelResult) -> dmc.Stack:
    default_channel = result.channels[0]
    return dmc.Stack(
        gap="lg",
        children=[
            page_header(
                "Response Curves",
                "Saturation curves from fitted geometric adstock + logistic saturation. "
                "The shaded band is the 94% posterior HDI.",
            ),
            dmc.Group(
                justify="space-between",
                align="end",
                children=[
                    dmc.Select(
                        label="Channel",
                        id=CHANNEL_SELECT_ID,
                        data=[{"value": c, "label": c} for c in result.channels],
                        value=default_channel,
                        w=240,
                        checkIconPosition="right",
                    ),
                    dmc.Text(
                        "Markers: average weekly spend, observed P5-P95 support, and spend where "
                        "logistic saturation reaches 90% of its asymptote.",
                        size="xs",
                        c="dimmed",
                        maw=560,
                        ta="right",
                    ),
                ],
            ),
            section(
                "Saturation Curve",
                "Expected incremental weekly revenue as a function of sustained weekly spend.",
                dmc.Stack(
                    gap="md",
                    children=[
                        dmc.Box(id=CHANNEL_STATS_ID, children=response_stats(result, default_channel)),
                        dcc.Graph(
                            id=CHANNEL_GRAPH_ID,
                            figure=response_curve_figure(result, default_channel),
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
            ),
        ],
    )


def register_response_curve_callbacks(app, results_by_geo: dict[str, ModelResult]) -> None:
    @app.callback(
        Output(CHANNEL_GRAPH_ID, "figure"),
        Output(CHANNEL_STATS_ID, "children"),
        Input(CHANNEL_SELECT_ID, "value"),
        Input(MODEL_REFRESH_STORE, "data"),
    )
    def _update(channel: str, _refresh: int | None):
        result = results_by_geo["All"]
        if not channel:
            channel = result.channels[0]
        return response_curve_figure(result, channel), response_stats(result, channel)

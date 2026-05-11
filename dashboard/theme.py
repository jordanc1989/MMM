"""Shared Plotly and Mantine theme values for the MMM dashboard."""

from __future__ import annotations

import plotly.graph_objects as go

from data.loader import CHANNELS

FONT_FAMILY = "DM Sans, sans-serif"

PALETTE: list[str] = [
    "#14b8a6",  # teal
    "#38bdf8",  # sky
    "#a78bfa",  # violet
    "#f59e0b",  # amber
    "#ef4444",  # rose
    "#22c55e",  # green
]
"""Consistent 6-colour palette used across the app."""

DARK_SCALE: list[str] = [
    "#d4d6d9",
    "#b1b4ba",
    "#8a8f97",
    "#62666d",
    "#3a3f47",
    "#2a2f36",
    "#23272e",
    "#181c21",
    "#13161a",
    "#0b0d10",
]

CHANNEL_COLORS: dict[str, str] = {c: PALETTE[i] for i, c in enumerate(CHANNELS)}

CHART_FONT_COLOR = DARK_SCALE[0]
GRID_COLOR = DARK_SCALE[6]
ZERO_LINE_COLOR = DARK_SCALE[5]
SURFACE_COLOR = DARK_SCALE[7]


def mantine_theme() -> dict:
    """Return the Mantine provider theme using the dashboard palette."""
    return {
        "primaryColor": "teal",
        "fontFamily": FONT_FAMILY,
        "headings": {"fontFamily": FONT_FAMILY, "fontWeight": "600"},
        "defaultRadius": "md",
        "colors": {
            "dark": DARK_SCALE,
        },
    }


def apply_dark_theme(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Apply the project-wide dark, transparent, DM Sans theme to a figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=CHART_FONT_COLOR, size=12),
        margin=dict(l=16, r=16, t=32, b=16),
        colorway=PALETTE,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=CHART_FONT_COLOR),
        ),
        hoverlabel=dict(
            bgcolor=SURFACE_COLOR,
            bordercolor=GRID_COLOR,
            font=dict(family=FONT_FAMILY, color=CHART_FONT_COLOR),
        ),
    )
    if height is not None:
        fig.update_layout(height=height)

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=GRID_COLOR,
        tickcolor=GRID_COLOR,
        tickfont=dict(color=CHART_FONT_COLOR),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        zeroline=True,
        zerolinecolor=ZERO_LINE_COLOR,
        linecolor=GRID_COLOR,
        tickcolor=GRID_COLOR,
        tickfont=dict(color=CHART_FONT_COLOR),
    )
    return fig


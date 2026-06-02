"""Shared Plotly and Mantine theme values for the MMM dashboard.

Editorial-dark system: a warm neutral ink, one restrained ochre accent, a muted
(non-neon) categorical palette for channels, and a serif/sans/mono type stack. A
small set of semantic colours (accent / positive / caution / neutral data ink)
keeps every figure and table reading from the same vocabulary.
"""

from __future__ import annotations

import plotly.graph_objects as go

from data.loader import CHANNELS

# ---- type stack ----------------------------------------------------------
FONT_FAMILY = "IBM Plex Sans, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
HEADING_FONT = "Source Serif 4, Georgia, serif"
MONO_FONT = "IBM Plex Mono, ui-monospace, monospace"

APP_TITLE = "Bayesian MMM Dashboard"

# ---- channel palette (muted, perceptually spaced — replaces the neon set) -
PALETTE: list[str] = [
    "#6f9170",  # sage
    "#c79141",  # ochre
    "#7e72a6",  # plum
    "#4f7f9c",  # steel
    "#b06a5e",  # clay
    "#9a8c5a",  # olive
]
"""Consistent muted 6-colour palette used across the app."""

# ---- semantic colours ----------------------------------------------------
ACCENT = "#c79141"  # ochre — the single signature accent (fit line, emphasis)
POSITIVE_COLOR = "#6f9170"  # sage — gains, in-support, "good"
NEGATIVE_COLOR = "#b06a5e"  # clay — losses, out-of-support, caution
STEEL_COLOR = "#4f7f9c"  # secondary series / totals

# Mantine 10-shade scale for the ochre accent (registered as `primaryColor`).
OCHRE_SCALE: list[str] = [
    "#faf3e6",
    "#f1e3c6",
    "#e6cd9c",
    "#dab873",
    "#d0a753",
    "#c79141",
    "#b07d37",
    "#8f662d",
    "#6e4e23",
    "#4d3718",
]

# Warm neutral ramp (light -> dark) backing both Mantine `dark` and chart greys.
DARK_SCALE: list[str] = [
    "#e6e4df",  # 0 text
    "#cbc8c1",  # 1
    "#a3a09a",  # 2 muted
    "#7a7872",  # 3 data ink (quiet)
    "#56544f",  # 4
    "#3a3934",  # 5 zero line
    "#2c2c31",  # 6 grid / border
    "#1d1d21",  # 7 surface
    "#17171a",  # 8
    "#111113",  # 9 background
]

CHANNEL_COLORS: dict[str, str] = {c: PALETTE[i] for i, c in enumerate(CHANNELS)}

CHART_FONT_COLOR = DARK_SCALE[0]
DATA_INK = DARK_SCALE[3]
GRID_COLOR = DARK_SCALE[6]
ZERO_LINE_COLOR = DARK_SCALE[5]
SURFACE_COLOR = DARK_SCALE[7]
PAGE_MAX_WIDTH = "1440px"
APP_MIN_HEIGHT_STYLE = {"minHeight": "100vh"}
PAGE_CONTAINER_STYLE = {"maxWidth": PAGE_MAX_WIDTH, "margin": "0 auto"}
OVERVIEW_TOOLBAR_STYLE = {
    "maxWidth": PAGE_MAX_WIDTH,
    "margin": "0 auto 16px auto",
}
REFIT_OVERLAY_STYLE = {
    "position": "fixed",
    "inset": 0,
    "zIndex": 10000,
    "alignItems": "center",
    "justifyContent": "center",
    "flexDirection": "column",
    "backgroundColor": "rgba(10, 11, 13, 0.72)",
    "backdropFilter": "blur(8px)",
}


def with_alpha(hex_color: str, alpha: float) -> str:
    """Return ``rgba(...)`` for a ``#rrggbb`` colour at the given alpha."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def mantine_theme() -> dict:
    """Return the Mantine provider theme using the editorial-dark palette."""
    return {
        "primaryColor": "ochre",
        "primaryShade": {"light": 6, "dark": 5},
        "autoContrast": True,
        "fontFamily": FONT_FAMILY,
        "fontFamilyMonospace": MONO_FONT,
        "headings": {"fontFamily": HEADING_FONT, "fontWeight": "600"},
        "defaultRadius": "sm",
        "colors": {
            "dark": DARK_SCALE,
            "ochre": OCHRE_SCALE,
        },
    }


def apply_dark_theme(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Apply the project-wide dark, transparent, editorial theme to a figure."""
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
            font=dict(size=11, color=DARK_SCALE[2]),
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
        tickfont=dict(color=DARK_SCALE[2]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        zeroline=True,
        zerolinecolor=ZERO_LINE_COLOR,
        linecolor=GRID_COLOR,
        tickcolor=GRID_COLOR,
        tickfont=dict(color=DARK_SCALE[2]),
    )
    return fig

"""Response curves page layout."""

from __future__ import annotations

import dash_mantine_components as dmc
import numpy as np
from dash import dcc

from components import fmt_currency, page_header, section
from figures.response_curves import response_curve_figure
from model.mmm import ModelResult, observed_spend_support, response_curve

CHANNEL_SELECT_ID = "response-curve-channel"
CHANNEL_GRAPH_ID = "response-curve-graph"
CHANNEL_STATS_ID = "response-curve-stats"


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
        ("Avg weekly spend", fmt_currency(current)),
        ("Current weekly contribution", fmt_currency(cur_weekly_rev)),
        ("Marginal ROAS at avg spend", f"{marginal_roas:.2f}x"),
        ("Observed spend range", f"{fmt_currency(support_p5)}–{fmt_currency(support_p95)}"),
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
                        "logistic saturation reaches 90% of its asymptote when that point is in range.",
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

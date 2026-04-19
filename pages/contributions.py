"""Channel Contributions page: stacked area, contribution %, per-channel table.

The stacked-area HDI band shows posterior uncertainty for the *sum of paid channel
contributions* only. Baseline (intercept, trend, seasonality, controls) is drawn at
posterior mean, so the band is not a full predictive envelope for total revenue.
"""

from __future__ import annotations

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc
from dash.exceptions import PreventUpdate

from components import CHANNEL_COLORS, apply_dark_theme, page_header, section
from components.ids import MODEL_REFRESH_STORE
from model.mmm import (
    ModelResult,
    channel_window_roi_hdi,
    marginal_slope_roas,
    paid_increment_hdi_arrays,
    slice_model_result,
)

CONTRIBUTION_STACK_GRAPH_ID = "contributions-stack-graph"
CONTRIBUTION_SHARE_GRAPH_ID = "contributions-share-graph"
CONTRIBUTIONS_ROI_GRAPH_ID = "contributions-roi-graph"
CONTRIBUTIONS_TABLE_ID = "contributions-table"


# ---------- charts ---------------------------------------------------------


def _cumulative_stack_edges(result: ModelResult) -> list[np.ndarray]:
    """Upper boundaries of each stacked layer (baseline, then each channel)."""
    edges: list[np.ndarray] = [np.asarray(result.baseline, dtype=float)]
    for ch in result.channels:
        edges.append(edges[-1] + np.asarray(result.contributions[ch], dtype=float))
    return edges


def contributions_area_chart(result: ModelResult) -> go.Figure:
    """Stack posterior-mean baseline and channels; optional 94% band on paid total only."""
    dates = pd.to_datetime(result.dates)
    edges = _cumulative_stack_edges(result)
    fig = go.Figure()

    paid_bounds = paid_increment_hdi_arrays(result)
    if paid_bounds is not None:
        # Band = posterior mean baseline + paid-media quantiles (baseline uncertainty omitted).
        pl, ph = paid_bounds
        upper = np.asarray(result.baseline, dtype=float) + ph
        lower = np.asarray(result.baseline, dtype=float) + pl
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=upper,
                mode="lines",
                line=dict(width=0),
                name="Paid media 94% HDI",
                legendgroup="hdi",
                hovertemplate="Upper (paid total) %{y:$,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(45, 212, 191, 0.12)",
                name="",
                legendgroup="hdi",
                showlegend=False,
                hovertemplate="Lower (paid total) %{y:$,.0f}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=edges[0],
            mode="lines",
            name="Baseline",
            line=dict(width=0),
            fillcolor="rgba(138, 143, 151, 0.25)",
            fill="tozeroy",
            hovertemplate="Baseline: $%{customdata:,.0f}<extra></extra>",
            customdata=edges[0],
        )
    )
    for i, channel in enumerate(result.channels):
        inc = edges[i + 1] - edges[i]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=edges[i + 1],
                mode="lines",
                name=channel,
                line=dict(width=0),
                fillcolor=CHANNEL_COLORS[channel],
                fill="tonexty",
                hovertemplate=f"{channel}: $%{{customdata:,.0f}}<extra></extra>",
                customdata=inc,
            )
        )

    fig.update_layout(
        hovermode="x unified",
        yaxis_tickprefix="$",
        yaxis_tickformat=".2s",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return apply_dark_theme(fig, height=380)


def contribution_share_bar(result: ModelResult) -> go.Figure:
    totals = result.total_contribution
    paid_total = sum(totals.values()) or 1.0
    rows = sorted(
        [(c, totals[c], totals[c] / paid_total) for c in result.channels],
        key=lambda r: r[1],
    )
    channels = [r[0] for r in rows]
    values = [r[2] * 100 for r in rows]
    labels = [f"{v:.1f}%" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=channels,
            orientation="h",
            text=labels,
            textposition="outside",
            marker=dict(color=[CHANNEL_COLORS[c] for c in channels]),
            hovertemplate="%{y}: %{x:.1f}%% of paid contribution<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(ticksuffix="%", range=[0, max(values) * 1.25]),
        showlegend=False,
    )
    fig.update_yaxes(showgrid=False)
    return apply_dark_theme(fig, height=380)


def roi_vs_marginal_chart(result: ModelResult) -> go.Figure:
    """Grouped bars: window ROAS (with 94% HDI) vs marginal ROAS (slope at mean spend)."""
    chans = result.channels
    means, err_hi, err_lo = [], [], []
    slopes = []
    for c in chans:
        m, lo, hi = channel_window_roi_hdi(result, c)
        means.append(m)
        err_lo.append(max(0.0, m - lo))
        err_hi.append(max(0.0, hi - m))
        slopes.append(marginal_slope_roas(result, c))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Window ROAS (mean ± 94% HDI)",
            x=chans,
            y=means,
            marker_color="#2dd4bf",
            error_y=dict(
                type="data",
                symmetric=False,
                array=err_hi,
                arrayminus=err_lo,
                thickness=1.2,
                color="#94a3b8",
            ),
            hovertemplate="%{x}<br>ROAS %{y:.2f}x<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Marginal ROAS (slope)",
            x=chans,
            y=slopes,
            marker_color="#38bdf8",
            hovertemplate="%{x}<br>Marginal %{y:.2f}x<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="ROAS factor (revenue per $ spend)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(rangemode="tozero")
    return apply_dark_theme(fig, height=400)


# ---------- table ----------------------------------------------------------


def _fmt_currency(v: float) -> str:
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:.1f}K"
    return f"${v:,.0f}"


def channel_table(result: ModelResult) -> dmc.Table:
    rows = []
    paid_total = sum(result.total_contribution.values()) or 1.0
    for c in result.channels:
        spend = result.total_spend[c]
        contrib = result.total_contribution[c]
        roi = contrib / spend if spend > 0 else 0.0
        share = contrib / paid_total * 100
        rows.append(
            dmc.TableTr(
                [
                    dmc.TableTd(
                        dmc.Group(
                            gap="sm",
                            children=[
                                dmc.Box(
                                    w=10,
                                    h=10,
                                    bg=CHANNEL_COLORS[c],
                                    style={"borderRadius": "2px"},
                                ),
                                dmc.Text(c, fw=500, size="sm"),
                            ],
                        )
                    ),
                    dmc.TableTd(
                        dmc.Text(_fmt_currency(spend), size="sm", className="mmm-numeric"),
                    ),
                    dmc.TableTd(
                        dmc.Text(_fmt_currency(contrib), size="sm", className="mmm-numeric"),
                    ),
                    dmc.TableTd(
                        dmc.Text(f"{share:.1f}%", size="sm", className="mmm-numeric"),
                    ),
                    dmc.TableTd(
                        dmc.Text(
                            f"{roi:.2f}x",
                            size="sm",
                            fw=600,
                            c="teal" if roi >= 1 else "orange",
                            className="mmm-numeric",
                        ),
                    ),
                ]
            )
        )

    head = dmc.TableThead(
        dmc.TableTr(
            [
                dmc.TableTh("Channel"),
                dmc.TableTh("Spend"),
                dmc.TableTh("Attributed Revenue"),
                dmc.TableTh("Share"),
                dmc.TableTh("ROI"),
            ]
        )
    )
    return dmc.Table(
        children=[head, dmc.TableTbody(rows)],
        striped=True,
        highlightOnHover=True,
        withTableBorder=False,
        verticalSpacing="sm",
    )


# ---------- layout ---------------------------------------------------------


def build_contributions(result: ModelResult) -> dmc.Stack:
    return dmc.Stack(
        gap="lg",
        children=[
            page_header(
                "Channel Contributions",
                "How each paid channel builds revenue over time.",
            ),
            section(
                "Weekly Contribution Stack",
                "Baseline at posterior mean, posterior-mean channel layers, and a 94% band on "
                "total paid media only (uncertainty in baseline/controls/seasonality not shown).",
                dcc.Graph(
                    id=CONTRIBUTION_STACK_GRAPH_ID,
                    figure=contributions_area_chart(result),
                    config={"displayModeBar": False},
                ),
            ),
            section(
                "Window ROAS vs marginal ROAS",
                "Window ROAS uses summed posterior channel contributions vs observed spend (bars show 94% HDI). "
                "Marginal ROAS is the slope of the steady-state response curve at mean weekly spend.",
                dcc.Graph(
                    id=CONTRIBUTIONS_ROI_GRAPH_ID,
                    figure=roi_vs_marginal_chart(result),
                    config={"displayModeBar": False},
                ),
            ),
            dmc.Grid(
                gutter="md",
                children=[
                    dmc.GridCol(
                        section(
                            "Share of Paid Contribution",
                            "Attributed revenue share across the full fitted period.",
                            dcc.Graph(
                                id=CONTRIBUTION_SHARE_GRAPH_ID,
                                figure=contribution_share_bar(result),
                                config={"displayModeBar": False},
                            ),
                        ),
                        span={"base": 12, "lg": 6},
                    ),
                    dmc.GridCol(
                        section(
                            "Per-Channel Summary",
                            "Spend, attributed revenue and ROI for the full fitted period.",
                            dmc.Box(id=CONTRIBUTIONS_TABLE_ID, children=channel_table(result)),
                        ),
                        span={"base": 12, "lg": 6},
                    ),
                ],
            ),
        ],
    )


def register_contributions_callbacks(app, results_by_geo: dict[str, ModelResult]) -> None:
    @app.callback(
        Output(CONTRIBUTION_STACK_GRAPH_ID, "figure"),
        Output(CONTRIBUTIONS_ROI_GRAPH_ID, "figure"),
        Output(CONTRIBUTION_SHARE_GRAPH_ID, "figure"),
        Output(CONTRIBUTIONS_TABLE_ID, "children"),
        Input("url", "pathname"),
        Input(MODEL_REFRESH_STORE, "data"),
    )
    def _update_contributions(pathname: str | None, _refresh: int | None):
        pathname = pathname or "/"
        if pathname != "/contributions":
            raise PreventUpdate
        base = results_by_geo["All"]
        dmin = pd.to_datetime(base.dates).min()
        dmax = pd.to_datetime(base.dates).max()
        sl = slice_model_result(base, dmin, dmax)
        return (
            contributions_area_chart(sl),
            roi_vs_marginal_chart(sl),
            contribution_share_bar(sl),
            channel_table(sl),
        )

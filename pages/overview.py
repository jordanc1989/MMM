"""Overview page: model health KPIs, actual vs predicted, revenue waterfall, residuals."""

from __future__ import annotations

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc
from plotly.subplots import make_subplots

from components.ids import GLOBAL_DATE_RANGE
from components import (
    CHANNEL_COLORS,
    apply_dark_theme,
    kpi_card,
    page_header,
    section,
)
from model.mmm import (
    ModelResult,
    posterior_predictive_interval_for_result,
    slice_model_result,
)

ACTUAL_PREDICTED_GRAPH_ID = "overview-actual-predicted-graph"
ACTUAL_PREDICTED_CLIP_ID = "overview-actual-predicted-clip"
OVERVIEW_KPI_GRID_ID = "overview-kpis"
OVERVIEW_WATERFALL_ID = "overview-waterfall"
OVERVIEW_RESIDUALS_ID = "overview-residuals"


# ---------- charts ---------------------------------------------------------


def actual_vs_predicted_chart(result: ModelResult, *, clip_y: bool = False) -> go.Figure:
    dates = pd.to_datetime(result.dates)
    fig = go.Figure()

    pp = posterior_predictive_interval_for_result(result)
    if pp is not None and len(pp) > 0:
        pdates = pd.to_datetime(pp["date"])
        upper = pp["abs_error_94_upper"].to_numpy(dtype=float)
        lower = pp["abs_error_94_lower"].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=pdates,
                y=upper,
                mode="lines",
                line=dict(width=0),
                name="94% posterior predictive",
                legendgroup="pp",
                showlegend=True,
                hovertemplate="%{x|%b %d, %Y}<br>upper %{y:$,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pdates,
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(45, 212, 191, 0.14)",
                name="",
                legendgroup="pp",
                showlegend=False,
                hovertemplate="%{x|%b %d, %Y}<br>lower %{y:$,.0f}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=result.revenue,
            mode="lines",
            name="Actual",
            line=dict(color="#64748b", width=2, shape="spline", smoothing=0.3),
            fill="tozeroy",
            fillcolor="rgba(100, 116, 139, 0.14)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=result.fitted,
            mode="lines",
            name="Posterior mean fit",
            line=dict(color="#2dd4bf", width=2.5, dash="dot", shape="spline", smoothing=0.3),
        )
    )
    fig.update_layout(
        hovermode="x unified",
        yaxis_tickprefix="$",
        yaxis_tickformat=".2s",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if clip_y:
        y_min = float(min(result.revenue.min(), result.fitted.min()))
        y_max = float(max(result.revenue.max(), result.fitted.max()))
        if pp is not None and len(pp) > 0:
            y_min = float(min(y_min, pp["abs_error_94_lower"].min()))
            y_max = float(max(y_max, pp["abs_error_94_upper"].max()))
        lower = max(0.0, y_min * 0.98)
        upper = y_max * 1.02
        fig.update_yaxes(range=[lower, upper])
    return apply_dark_theme(fig, height=360)


def revenue_waterfall(result: ModelResult) -> go.Figure:
    """Decompose total revenue into Intercept / Trend / Seasonality / Controls / Channels."""
    intercept_total = float(result.intercept_contribution.sum())
    trend_total = float(result.trend_contribution.sum())
    seasonality_total = float(result.seasonality_contribution.sum())
    controls_total = float(
        np.sum([series.sum() for series in result.control_contributions.values()])
    )
    channel_totals = {c: float(result.contributions[c].sum()) for c in result.channels}
    predicted_total = (
        intercept_total
        + trend_total
        + seasonality_total
        + controls_total
        + sum(channel_totals.values())
    )

    labels = [
        "Intercept",
        "Trend",
        "Seasonality",
        "Controls",
        *result.channels,
        "Predicted Revenue",
    ]
    measures = (
        ["absolute"]
        + ["relative"] * 3
        + ["relative"] * len(result.channels)
        + ["total"]
    )
    values = [
        intercept_total,
        trend_total,
        seasonality_total,
        controls_total,
        *[channel_totals[c] for c in result.channels],
        predicted_total,
    ]

    fig = go.Figure(
        go.Waterfall(
            measure=measures,
            x=labels,
            y=values,
            text=[f"${v/1e6:,.1f}M" for v in values],
            textposition="outside",
            connector=dict(line=dict(color="#3a3f47", width=1)),
            increasing=dict(marker=dict(color="#14b8a6")),
            decreasing=dict(marker=dict(color="#ef4444")),
            totals=dict(marker=dict(color="#38bdf8")),
        )
    )
    fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=".2s", showlegend=False)
    fig.update_xaxes(tickangle=-25)
    return apply_dark_theme(fig, height=360)


def _acf(x: np.ndarray, n_lags: int) -> np.ndarray:
    """Simple biased autocorrelation for lags 1..n_lags."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.zeros(n_lags)
    out = np.empty(n_lags)
    n = len(x)
    for lag in range(1, n_lags + 1):
        out[lag - 1] = float(np.dot(x[: n - lag], x[lag:])) / denom
    return out


def residuals_diagnostic_figure(result: ModelResult) -> go.Figure:
    """Two-row diagnostic: residuals vs time (top) + lag 1-12 ACF (bottom)."""
    dates = pd.to_datetime(result.dates)
    residuals = result.residuals
    n_lags = min(12, max(1, len(residuals) // 4))
    acf_vals = _acf(residuals, n_lags)
    band = 1.96 / max(np.sqrt(len(residuals)), 1.0)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.62, 0.38],
        shared_xaxes=False,
        vertical_spacing=0.14,
        subplot_titles=("Residuals vs time", f"Autocorrelation (lags 1–{n_lags})"),
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=residuals,
            mode="lines",
            name="Residuals",
            line=dict(color="#2dd4bf", width=1.5, shape="spline", smoothing=0.3),
            fill="tozeroy",
            fillcolor="rgba(45, 212, 191, 0.18)",
            hovertemplate="%{x|%b %d, %Y}<br>%{y:$,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0,
        line=dict(color="#3a3f47", width=1, dash="dot"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=list(range(1, n_lags + 1)),
            y=acf_vals,
            name="ACF",
            marker=dict(color="#38bdf8"),
            hovertemplate="Lag %{x}<br>ρ=%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=band,
        line=dict(color="#64748b", width=1, dash="dash"),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=-band,
        line=dict(color="#64748b", width=1, dash="dash"),
        row=2,
        col=1,
    )

    fig.update_yaxes(tickprefix="$", tickformat=".2s", row=1, col=1)
    fig.update_yaxes(range=[-1.0, 1.0], row=2, col=1)
    fig.update_xaxes(title=None, row=1, col=1)
    fig.update_xaxes(title="Lag (weeks)", row=2, col=1)
    fig.update_layout(showlegend=False)
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(family="DM Sans, sans-serif", color="#d4d6d9", size=13)

    return apply_dark_theme(fig, height=360)


# ---------- layout helpers -------------------------------------------------


def _fmt_currency(v: float) -> str:
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:.1f}K"
    return f"${v:,.0f}"


def _build_kpi_grid(result: ModelResult) -> dmc.SimpleGrid:
    dates = pd.to_datetime(result.dates)
    date_range = f"{dates.min():%b %Y} – {dates.max():%b %Y}"
    total_rev = result.total_revenue
    paid_share = sum(result.total_contribution.values()) / total_rev if total_rev else 0.0
    lo, hi = result.r2_hdi

    total_spend = float(sum(result.total_spend.values()))
    attrib_paid = float(sum(result.total_contribution.values()))
    blended = attrib_paid / total_spend if total_spend > 0 else 0.0
    top_ch = (
        max(result.channels, key=lambda c: result.roi[c]) if result.channels else "—"
    )

    return dmc.SimpleGrid(
        cols={"base": 1, "sm": 2, "lg": 4},
        spacing="md",
        children=[
            kpi_card(
                label="Model R²",
                value=f"{result.r2:.2f}",
                icon="tabler:chart-dots",
                helper="Explained variance on weekly revenue",
                sub=f"94% HDI [{lo:.2f}, {hi:.2f}]",
            ),
            kpi_card(
                label="MAPE",
                value=f"{result.mape*100:.1f}%",
                icon="tabler:target",
                helper="Mean absolute percent error",
            ),
            kpi_card(
                label="Total revenue (observed)",
                value=_fmt_currency(total_rev),
                icon="tabler:currency-dollar",
                helper=f"Paid-media share of observed {paid_share*100:.1f}%",
            ),
            kpi_card(
                label="Weeks in view",
                value=str(result.n_weeks),
                icon="tabler:calendar",
                helper=date_range,
            ),
            kpi_card(
                label="Total media spend",
                value=_fmt_currency(total_spend),
                icon="tabler:credit-card",
                helper="Sum of paid channel spend in period",
            ),
            kpi_card(
                label="Attributed revenue (paid)",
                value=_fmt_currency(attrib_paid),
                icon="tabler:chart-area",
                helper="Posterior mean channel contributions",
            ),
            kpi_card(
                label="Blended ROAS",
                value=f"{blended:.2f}x",
                icon="tabler:scale",
                helper="Attributed paid revenue ÷ total spend",
            ),
            kpi_card(
                label="Top channel (ROAS)",
                value=top_ch,
                icon="tabler:crown",
                helper="Highest window ROAS in selection",
            ),
        ],
    )


def build_overview(result: ModelResult) -> dmc.Stack:
    charts = dmc.Grid(
        gutter="md",
        children=[
            dmc.GridCol(
                section(
                    "Actual vs Predicted Revenue",
                    "Weekly observed revenue, posterior mean fit, and 94% posterior predictive band.",
                    dmc.Stack(
                        gap="sm",
                        children=[
                            dmc.Group(
                                justify="flex-end",
                                children=[
                                    dmc.Switch(
                                        id=ACTUAL_PREDICTED_CLIP_ID,
                                        label="Clip y-axis",
                                        size="sm",
                                        checked=False,
                                        color="teal",
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                id=ACTUAL_PREDICTED_GRAPH_ID,
                                figure=actual_vs_predicted_chart(result),
                                config={"displayModeBar": False},
                            ),
                        ],
                    ),
                ),
                span={"base": 12, "lg": 7},
            ),
            dmc.GridCol(
                section(
                    "Revenue Decomposition",
                    "Intercept, trend, seasonality, controls, and channel lift for the selected window.",
                    dcc.Graph(
                        id=OVERVIEW_WATERFALL_ID,
                        figure=revenue_waterfall(result),
                        config={"displayModeBar": False},
                    ),
                ),
                span={"base": 12, "lg": 5},
            ),
        ],
    )

    diagnostics = section(
        "Residual Diagnostics",
        "Residuals and weekly autocorrelation — a well-specified model leaves noise near zero with small, non-structural ACF.",
        dcc.Graph(
            id=OVERVIEW_RESIDUALS_ID,
            figure=residuals_diagnostic_figure(result),
            config={"displayModeBar": False},
        ),
    )

    return dmc.Stack(
        gap="lg",
        children=[
            page_header(
                "Model Overview",
                "All geographic markets aggregated · weekly media-mix model with "
                f"{len(result.channels)} paid channels.",
            ),
            dmc.Box(id=OVERVIEW_KPI_GRID_ID, children=_build_kpi_grid(result)),
            charts,
            diagnostics,
        ],
    )


_ = CHANNEL_COLORS


def register_overview_callbacks(app, results_by_geo: dict[str, ModelResult]) -> None:
    base = results_by_geo["All"]

    @app.callback(
        Output(ACTUAL_PREDICTED_GRAPH_ID, "figure"),
        Output(OVERVIEW_KPI_GRID_ID, "children"),
        Output(OVERVIEW_WATERFALL_ID, "figure"),
        Output(OVERVIEW_RESIDUALS_ID, "figure"),
        Input(GLOBAL_DATE_RANGE, "start_date"),
        Input(GLOBAL_DATE_RANGE, "end_date"),
        Input(ACTUAL_PREDICTED_CLIP_ID, "checked"),
    )
    def _update_overview(start, end, clip_y):
        dmin = pd.to_datetime(base.dates).min()
        dmax = pd.to_datetime(base.dates).max()
        if start is None:
            start = dmin
        else:
            start = pd.Timestamp(start)
        if end is None:
            end = dmax
        else:
            end = pd.Timestamp(end)
        start = max(start, dmin)
        end = min(end, dmax)
        if start > end:
            start, end = end, start

        sliced = slice_model_result(base, start, end)
        fig_pred = actual_vs_predicted_chart(sliced, clip_y=bool(clip_y))
        kpis = _build_kpi_grid(sliced)
        fig_wf = revenue_waterfall(sliced)
        fig_res = residuals_diagnostic_figure(sliced)
        return fig_pred, kpis, fig_wf, fig_res

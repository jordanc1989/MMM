"""Overview page Plotly figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.theme import (
    ACCENT,
    CHART_FONT_COLOR,
    DATA_INK,
    FONT_FAMILY,
    GRID_COLOR,
    NEGATIVE_COLOR,
    STEEL_COLOR,
    apply_dark_theme,
    with_alpha,
)
from data.loader import TREND_COLUMNS
from model.mmm import ModelResult, posterior_predictive_interval_for_result

def actual_vs_predicted_chart(result: ModelResult) -> go.Figure:
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
                fillcolor=with_alpha(ACCENT, 0.12),
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
            line=dict(color=DATA_INK, width=1.8),
            fill="tozeroy",
            fillcolor=with_alpha(DATA_INK, 0.10),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=result.fitted,
            mode="lines",
            name="Posterior mean fit",
            line=dict(color=ACCENT, width=2.2, dash="dot"),
        )
    )
    fig.update_layout(
        hovermode="x unified",
        yaxis_tickprefix="$",
        yaxis_tickformat=".2s",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return apply_dark_theme(fig, height=360)


def revenue_waterfall(result: ModelResult) -> go.Figure:
    """Decompose total revenue into configured baseline terms and channels."""
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

    include_trend = bool(TREND_COLUMNS) or abs(trend_total) > 1e-6
    labels = ["Intercept"]
    values = [intercept_total]
    if include_trend:
        labels.append("Trend")
        values.append(trend_total)
    labels.extend(["Seasonality", "Controls", *result.channels, "Predicted revenue"])
    values.extend(
        [
            seasonality_total,
            controls_total,
            *[channel_totals[c] for c in result.channels],
            predicted_total,
        ]
    )
    measures = ["absolute"] + ["relative"] * (len(values) - 2) + ["total"]

    fig = go.Figure(
        go.Waterfall(
            measure=measures,
            x=labels,
            y=values,
            text=[f"${v/1e6:,.1f}M" for v in values],
            textposition="outside",
            connector=dict(line=dict(color=GRID_COLOR, width=1)),
            increasing=dict(marker=dict(color=ACCENT)),
            decreasing=dict(marker=dict(color=NEGATIVE_COLOR)),
            totals=dict(marker=dict(color=STEEL_COLOR)),
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
            line=dict(color=ACCENT, width=1.4),
            fill="tozeroy",
            fillcolor=with_alpha(ACCENT, 0.14),
            hovertemplate="%{x|%b %d, %Y}<br>%{y:$,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0,
        line=dict(color=GRID_COLOR, width=1, dash="dot"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=list(range(1, n_lags + 1)),
            y=acf_vals,
            name="ACF",
            marker=dict(color=STEEL_COLOR),
            hovertemplate="Lag %{x}<br>ρ=%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=band,
        line=dict(color=DATA_INK, width=1, dash="dash"),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=-band,
        line=dict(color=DATA_INK, width=1, dash="dash"),
        row=2,
        col=1,
    )

    fig.update_yaxes(tickprefix="$", tickformat=".2s", row=1, col=1)
    fig.update_yaxes(range=[-1.0, 1.0], row=2, col=1)
    fig.update_xaxes(title=None, row=1, col=1)
    fig.update_xaxes(title="Lag (weeks)", row=2, col=1)
    fig.update_layout(showlegend=False)
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(family=FONT_FAMILY, color=CHART_FONT_COLOR, size=13)

    return apply_dark_theme(fig, height=360)

"""Overview page: model health KPIs, actual vs predicted, revenue waterfall, residuals."""

from __future__ import annotations

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from components.ids import (
    MODEL_REFRESH_STORE,
    OVERVIEW_DATE_STORE,
    OVERVIEW_RANGE_PRESET,
    OVERVIEW_YEAR_SELECT,
    OVERVIEW_TOOLBAR,
)
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


def _fmt_opt(x: float | int | None) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4f}" if abs(x) < 10 else f"{x:.2f}"
    return str(int(x))


def mcmc_diagnostics_panel(result: ModelResult) -> dmc.Stack:
    """Collapsible content: MCMC summary (R-hat, ESS, divergences, energy, BFMI)."""
    d = result.mcmc_diagnostics
    if not d:
        return dmc.Stack(
            children=[
                dmc.Text(
                    "No MCMC diagnostics available.",
                    size="sm",
                    c="dimmed",
                ),
            ],
        )
    rows_fixed = [
        ("Divergences", _fmt_opt(d.get("divergences"))),
        ("Max R-hat", _fmt_opt(d.get("max_r_hat"))),
        ("Min ESS bulk", _fmt_opt(d.get("min_ess_bulk"))),
        ("Min ESS tail", _fmt_opt(d.get("min_ess_tail"))),
        ("BFMI (mean)", _fmt_opt(d.get("bfmi_mean"))),
        (
            "BFMI per chain",
            ", ".join(
                f"{x:.3f}" for x in (d.get("bfmi_per_chain") or [])
            )
            or "—",
        ),
        ("Energy mean", _fmt_opt(d.get("energy_mean"))),
        ("Energy SD", _fmt_opt(d.get("energy_sd"))),
        ("Chains", _fmt_opt(d.get("chains"))),
        ("Draws / chain", _fmt_opt(d.get("draws_per_chain"))),
    ]
    sc = result.sampler_config or {}
    rows_fixed.insert(
        0,
        ("Target accept", _fmt_opt(sc.get("target_accept"))),
    )
    rows_fixed.insert(1, ("Tune", _fmt_opt(sc.get("tune"))))
    rows_fixed.insert(2, ("Draws", _fmt_opt(sc.get("draws"))))
    rows_fixed.insert(3, ("Adstock l_max (lags)", str(result.adstock_l_max)))

    head = dmc.TableThead(
        dmc.TableTr(
            [
                dmc.TableTh("Metric", style={"width": "42%"}),
                dmc.TableTh("Value"),
            ]
        )
    )
    body = dmc.TableTbody(
        [
            dmc.TableTr([dmc.TableTd(a), dmc.TableTd(b)])
            for a, b in rows_fixed
        ]
    )
    children: list = []
    if result.half_life_truncation_warning:
        children.append(
            dmc.Alert(
                result.half_life_truncation_warning,
                color="yellow",
                title="Adstock truncation",
                variant="light",
            ),
        )
    children.extend(
        [
            dmc.Table(
                verticalSpacing="xs",
                highlightOnHover=True,
                withTableBorder=True,
                withColumnBorders=True,
                children=[head, body],
            ),
            dmc.Text(
                "BFMI is the Bayesian fraction of missing information in the energy "
                "distribution (higher is often better for exploration).",
                size="xs",
                c="dimmed",
            ),
        ]
    )
    return dmc.Stack(gap="sm", children=children)


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


def _bounds(base: ModelResult) -> tuple[pd.Timestamp, pd.Timestamp]:
    d = pd.to_datetime(base.dates)
    return pd.Timestamp(d.min()), pd.Timestamp(d.max())


def _range_from_preset(
    preset: str,
    year_str: str | None,
    dmin: pd.Timestamp,
    dmax: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if preset == "full":
        return dmin, dmax
    if preset == "l12m":
        start = dmax - pd.DateOffset(weeks=52)
        return max(dmin, pd.Timestamp(start)), dmax
    if preset == "l6m":
        start = dmax - pd.DateOffset(weeks=26)
        return max(dmin, pd.Timestamp(start)), dmax
    if preset == "year":
        y = int(year_str) if year_str is not None else int(dmax.year)
        ys = pd.Timestamp(year=y, month=1, day=1)
        ye = pd.Timestamp(year=y, month=12, day=31, hour=23, minute=59, second=59)
        return max(ys, dmin), min(ye, dmax)
    return dmin, dmax


def _try_prior_yoy_slice(
    base: ModelResult,
    sliced: ModelResult,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> ModelResult | None:
    """Prior-year window shifted back 52 weeks; require at least as many weeks as current."""
    dmin, dmax = _bounds(base)
    ps = start - pd.DateOffset(weeks=52)
    pe = end - pd.DateOffset(weeks=52)
    ps = max(ps, dmin)
    pe = min(pe, dmax)
    if ps > pe:
        return None
    prior = slice_model_result(base, ps, pe)
    if prior.n_weeks < sliced.n_weeks:
        return None
    return prior


def _yoy_pct_line(curr: float, prior: float) -> tuple[str | None, str]:
    if prior == 0:
        return None, "dimmed"
    pct = (curr - prior) / prior * 100.0
    s = f"YoY {'+' if pct >= 0 else ''}{pct:.1f}%"
    return s, "teal" if pct >= 0 else "red"


def _yoy_roas_line(curr: float, prior: float) -> tuple[str | None, str]:
    d = curr - prior
    if abs(d) < 1e-12 and curr == 0 and prior == 0:
        return None, "dimmed"
    s = f"YoY {'+' if d >= 0 else ''}{d:.2f}x"
    return s, "teal" if d >= 0 else "red"


def _build_kpi_grid(
    result: ModelResult,
    base: ModelResult,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dmc.SimpleGrid:
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

    prior = _try_prior_yoy_slice(base, result, start, end)
    yoy_rev = yoy_spend = yoy_attrib = yoy_roas = None
    c_rev = c_spend = c_attr = c_roas = "dimmed"
    if prior is not None:
        pr = float(prior.total_revenue)
        ps = float(sum(prior.total_spend.values()))
        pa = float(sum(prior.total_contribution.values()))
        pbl = pa / ps if ps > 0 else 0.0
        yoy_rev, c_rev = _yoy_pct_line(total_rev, pr)
        yoy_spend, c_spend = _yoy_pct_line(total_spend, ps)
        yoy_attrib, c_attr = _yoy_pct_line(attrib_paid, pa)
        yoy_roas, c_roas = _yoy_roas_line(blended, pbl)

    return dmc.SimpleGrid(
        cols={"base": 1, "sm": 2, "lg": 4},
        spacing="md",
        children=[
            kpi_card(
                label="Model R²",
                value=f"{result.r2:.2f}",
                icon="tabler:chart-dots",
                helper="Explained variance on weekly revenue",
                sub=f"Parameter 94% HDI [{lo:.2f}, {hi:.2f}]",
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
                yoy=yoy_rev,
                yoy_color=c_rev,
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
                yoy=yoy_spend,
                yoy_color=c_spend,
            ),
            kpi_card(
                label="Attributed revenue (paid)",
                value=_fmt_currency(attrib_paid),
                icon="tabler:chart-area",
                helper="Posterior mean channel contributions",
                yoy=yoy_attrib,
                yoy_color=c_attr,
            ),
            kpi_card(
                label="Blended ROAS",
                value=f"{blended:.2f}x",
                icon="tabler:scale",
                helper="Attributed paid revenue ÷ total spend",
                yoy=yoy_roas,
                yoy_color=c_roas,
            ),
            kpi_card(
                label="Top channel (ROAS)",
                value=top_ch,
                icon="tabler:crown",
                helper="Highest window ROAS in selection",
            ),
        ],
    )


def build_overview_toolbar(result: ModelResult) -> dmc.Box:
    """Mounted once in the app shell so preset/clip IDs always exist for callbacks."""
    dmin, dmax = _bounds(result)
    year_options = [
        {"value": str(y), "label": str(y)} for y in range(dmin.year, dmax.year + 1)
    ]
    default_year = str(int(dmax.year))

    range_controls = dmc.Paper(
        p="md",
        radius="md",
        shadow="sm",
        withBorder=False,
        className="mmm-paper",
        children=dmc.Group(
            gap="lg",
            align="flex-end",
            wrap="wrap",
            justify="space-between",
            children=[
                dmc.Group(
                    gap="lg",
                    align="flex-end",
                    wrap="wrap",
                    children=[
                        dmc.Stack(
                            gap=6,
                            children=[
                                dmc.Text(
                                    "Analysis window",
                                    size="xs",
                                    c="dimmed",
                                    tt="uppercase",
                                    fw=600,
                                ),
                                dmc.SegmentedControl(
                                    id=OVERVIEW_RANGE_PRESET,
                                    value="full",
                                    data=[
                                        {"value": "full", "label": "Full period"},
                                        {"value": "l12m", "label": "Last 12 mo"},
                                        {"value": "l6m", "label": "Last 6 mo"},
                                        {"value": "year", "label": "Year"},
                                    ],
                                    size="sm",
                                    color="teal",
                                    persistence=True,
                                    persistence_type="session",
                                ),
                            ],
                        ),
                        dmc.Select(
                            id=OVERVIEW_YEAR_SELECT,
                            data=year_options,
                            value=default_year,
                            w=140,
                            size="sm",
                            label="Calendar year",
                            style={"display": "none"},
                            clearable=False,
                            persistence=True,
                            persistence_type="session",
                        ),
                    ],
                ),
                dmc.Switch(
                    id=ACTUAL_PREDICTED_CLIP_ID,
                    label="Clip y-axis",
                    size="sm",
                    checked=False,
                    color="teal",
                ),
            ],
        ),
    )

    return dmc.Box(
        id=OVERVIEW_TOOLBAR,
        style={"maxWidth": "1440px", "margin": "0 auto 16px auto", "display": "block"},
        children=[range_controls],
    )


def build_overview(result: ModelResult) -> dmc.Stack:
    dmin, dmax = _bounds(result)

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

    diagnostics = dmc.Accordion(
        multiple=True,
        variant="separated",
        radius="md",
        chevronPosition="right",
        value=None,
        className="mmm-accordion-residuals",
        children=[
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        dmc.Stack(
                            gap=2,
                            children=[
                                dmc.Text("MCMC diagnostics", size="sm", fw=600),
                                dmc.Text(
                                    "R-hat, ESS, divergences, energy, BFMI, and sampler settings.",
                                    size="xs",
                                    c="dimmed",
                                ),
                            ],
                        )
                    ),
                    dmc.AccordionPanel(mcmc_diagnostics_panel(result)),
                ],
                value="mcmc",
            ),
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        dmc.Group(
                            justify="space-between",
                            align="center",
                            wrap="nowrap",
                            children=[
                                dmc.Stack(
                                    gap=2,
                                    children=[
                                        dmc.Text("Residual diagnostics", size="sm", fw=600),
                                        dmc.Text(
                                            "Residuals vs time and weekly ACF — expand to view.",
                                            size="xs",
                                            c="dimmed",
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ),
                    dmc.AccordionPanel(
                        dcc.Graph(
                            id=OVERVIEW_RESIDUALS_ID,
                            figure=residuals_diagnostic_figure(result),
                            config={"displayModeBar": False},
                        )
                    ),
                ],
                value="residuals",
            ),
        ],
    )

    return dmc.Stack(
        gap="lg",
        children=[
            page_header(
                "Model Overview",
                "All geographic markets aggregated · weekly media-mix model with "
                f"{len(result.channels)} paid channels.",
            ),
            dmc.Box(
                id=OVERVIEW_KPI_GRID_ID,
                children=_build_kpi_grid(result, result, dmin, dmax),
            ),
            charts,
            diagnostics,
        ],
    )


_ = CHANNEL_COLORS


def register_overview_callbacks(app, results_by_geo: dict[str, ModelResult]) -> None:
    @app.callback(
        Output(OVERVIEW_TOOLBAR, "style"),
        Input("url", "pathname"),
    )
    def _overview_toolbar_visibility(pathname: str | None):
        pathname = pathname or "/"
        base_style = {"maxWidth": "1440px", "margin": "0 auto 16px auto"}
        if pathname in ("/", ""):
            return {**base_style, "display": "block"}
        return {**base_style, "display": "none"}

    @app.callback(
        Output(OVERVIEW_DATE_STORE, "data"),
        Output(OVERVIEW_YEAR_SELECT, "style"),
        Input("url", "pathname"),
        Input(OVERVIEW_RANGE_PRESET, "value"),
        Input(OVERVIEW_YEAR_SELECT, "value"),
    )
    def _sync_range_from_presets(
        pathname: str | None, preset: str | None, year_sel: str | None
    ):
        pathname = pathname or "/"
        if pathname not in ("/", ""):
            raise PreventUpdate
        dmin, dmax = _bounds(results_by_geo["All"])
        p = preset or "full"
        s, e = _range_from_preset(p, year_sel, dmin, dmax)
        style = (
            {"display": "block", "minWidth": 140}
            if p == "year"
            else {"display": "none"}
        )
        return (
            {"start": s.date().isoformat(), "end": e.date().isoformat()},
            style,
        )

    @app.callback(
        Output(ACTUAL_PREDICTED_GRAPH_ID, "figure"),
        Output(OVERVIEW_KPI_GRID_ID, "children"),
        Output(OVERVIEW_WATERFALL_ID, "figure"),
        Output(OVERVIEW_RESIDUALS_ID, "figure"),
        Input("url", "pathname"),
        Input(OVERVIEW_DATE_STORE, "data"),
        Input(ACTUAL_PREDICTED_CLIP_ID, "checked"),
        Input(MODEL_REFRESH_STORE, "data"),
    )
    def _update_overview(pathname, data, clip_y, _refresh):
        pathname = pathname or "/"
        if pathname not in ("/", ""):
            raise PreventUpdate
        base = results_by_geo["All"]
        dmin = pd.to_datetime(base.dates).min()
        dmax = pd.to_datetime(base.dates).max()
        if not data or not isinstance(data, dict):
            start, end = dmin, dmax
        else:
            raw_s, raw_e = data.get("start"), data.get("end")
            if raw_s is None or raw_e is None:
                start, end = dmin, dmax
            else:
                start = pd.Timestamp(raw_s)
                end = pd.Timestamp(raw_e)
        start = max(start, dmin)
        end = min(end, dmax)
        if start > end:
            start, end = end, start

        sliced = slice_model_result(base, start, end)
        fig_pred = actual_vs_predicted_chart(sliced, clip_y=bool(clip_y))
        kpis = _build_kpi_grid(sliced, base, start, end)
        fig_wf = revenue_waterfall(sliced)
        fig_res = residuals_diagnostic_figure(sliced)
        return fig_pred, kpis, fig_wf, fig_res

"""Overview page layout and presentational helpers."""

from __future__ import annotations

import dash_mantine_components as dmc
import pandas as pd
from dash import dcc

from components import (
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    fmt_currency,
    fmt_pct,
    kpi_card,
    page_header,
    section,
)
from components.ids import (
    ACTUAL_PREDICTED_GRAPH_ID,
    OVERVIEW_DATE_STORE,
    OVERVIEW_KPI_GRID_ID,
    OVERVIEW_RANGE_PRESET,
    OVERVIEW_RESIDUALS_ID,
    OVERVIEW_TOOLBAR,
    OVERVIEW_WATERFALL_ID,
    OVERVIEW_YEAR_SELECT,
)
from dashboard.theme import OVERVIEW_TOOLBAR_STYLE
from data.loader import TREND_COLUMNS
from figures.overview import (
    actual_vs_predicted_chart,
    residuals_diagnostic_figure,
    revenue_waterfall,
)
from model.mmm import (
    ModelResult,
    rolling_origin_validation_metrics,
    slice_model_result,
)


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
        ("Cached draws / chain", _fmt_opt(d.get("draws_per_chain"))),
    ]
    sc = result.sampler_config or {}
    rows_fixed.insert(
        0,
        ("Target accept", _fmt_opt(sc.get("target_accept"))),
    )
    rows_fixed.insert(1, ("Requested tune / chain", _fmt_opt(sc.get("tune"))))
    rows_fixed.insert(2, ("Requested draws / chain", _fmt_opt(sc.get("draws"))))
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



# ---------- layout helpers -------------------------------------------------


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
    return s, POSITIVE_COLOR if pct >= 0 else NEGATIVE_COLOR


def _yoy_roas_line(curr: float, prior: float) -> tuple[str | None, str]:
    d = curr - prior
    if abs(d) < 1e-12 and curr == 0 and prior == 0:
        return None, "dimmed"
    s = f"YoY {'+' if d >= 0 else ''}{d:.2f}x"
    return s, POSITIVE_COLOR if d >= 0 else NEGATIVE_COLOR


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
    temporal = result.temporal_validation or {}
    recent_weeks = temporal.get("recent_weeks")
    recent_start = temporal.get("recent_start")

    total_spend = float(sum(result.total_spend.values()))
    attrib_paid = float(sum(result.total_contribution.values()))
    blended = attrib_paid / total_spend if total_spend > 0 else 0.0
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
                label="In-sample R²",
                value=f"{result.r2:.2f}",
                icon="tabler:chart-dots",
                helper="Explained variance on fitted weekly revenue",
                sub=f"Parameter 94% HDI [{lo:.2f}, {hi:.2f}]",
            ),
            kpi_card(
                label="In-sample MAPE",
                value=f"{result.mape*100:.1f}%",
                icon="tabler:target",
                helper="Mean absolute percent error on fitted period",
            ),
            kpi_card(
                label="Recent in-sample MAPE",
                value=fmt_pct(temporal.get("recent_mape")),
                icon="tabler:calendar-stats",
                helper=(
                    f"Last {recent_weeks} weeks from {recent_start}"
                    if recent_weeks and recent_start
                    else "Recent time block not available"
                ),
            ),
            kpi_card(
                label="Total revenue (observed)",
                value=fmt_currency(total_rev),
                icon="tabler:currency-dollar",
                helper=f"Paid-media share of observed {paid_share*100:.1f}%",
                yoy=yoy_rev,
                yoy_color=c_rev,
            ),
            kpi_card(
                label="Total media spend",
                value=fmt_currency(total_spend),
                icon="tabler:credit-card",
                helper="Sum of paid channel spend in period",
                yoy=yoy_spend,
                yoy_color=c_spend,
            ),
            kpi_card(
                label="Attributed revenue (paid)",
                value=fmt_currency(attrib_paid),
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
                label="Weeks in view",
                value=str(result.n_weeks),
                icon="tabler:calendar",
                helper=date_range,
            ),
        ],
    )


def backtest_panel(result: ModelResult) -> dmc.Stack:
    """Lightweight rolling-origin surrogate forecast check."""
    bt = rolling_origin_validation_metrics(result, horizon=4, n_cutoffs=3)
    rows = bt.get("rows") or []
    if not rows:
        return dmc.Stack(
            gap="sm",
            children=[
                dmc.Text(
                    "Not enough history for rolling-origin validation.",
                    size="sm",
                    c="dimmed",
                )
            ],
        )

    bias = bt.get("bias")
    bias_value = float(bias) if bias is not None else None
    if bias_value is None or abs(bias_value) < 0.0005:
        bias_direction = "no average directional bias"
    elif bias_value > 0:
        bias_direction = "predicted above actual"
    else:
        bias_direction = "actual above predicted"
    mape = float(bt.get("mape") or 0.0)
    coverage = float(bt.get("coverage") or 0.0)
    signal = (
        "The surrogate struggles on these holdouts"
        if mape >= 0.4 or abs(bias_value or 0.0) >= 0.15 or coverage < 0.9
        else "The surrogate is comparatively stable on these holdouts"
    )
    benchmark_rows = [
        dmc.TableTr(
            [
                dmc.TableTd(dmc.Text(str(row["method"]), size="sm")),
                dmc.TableTd(
                    dmc.Text(fmt_pct(row["mape"]), size="sm", className="mmm-numeric")
                ),
                dmc.TableTd(
                    dmc.Text(
                        fmt_currency(float(row["rmse"] or 0.0)),
                        size="sm",
                        className="mmm-numeric",
                    )
                ),
                dmc.TableTd(
                    dmc.Text(
                        fmt_pct(row["bias"]),
                        size="sm",
                        className="mmm-numeric",
                    )
                ),
                dmc.TableTd(
                    dmc.Text(
                        fmt_pct(row.get("coverage")),
                        size="sm",
                        className="mmm-numeric",
                    )
                ),
            ]
        )
        for row in (bt.get("benchmarks") or [])
    ]
    summary = dmc.SimpleGrid(
        cols={"base": 2, "md": 4},
        spacing="md",
        children=[
            dmc.Stack(
                gap=2,
                children=[
                    dmc.Text(label, size="xs", c="dimmed", tt="uppercase", fw=600),
                    dmc.Text(value, size="lg", fw=600, className="mmm-numeric"),
                ],
            )
            for label, value in [
                ("Surrogate MAPE", fmt_pct(bt.get("mape"))),
                ("Surrogate RMSE", fmt_currency(float(bt.get("rmse") or 0.0))),
                ("Forecast bias", fmt_pct(bt.get("bias"))),
                ("Surrogate interval coverage", fmt_pct(bt.get("coverage"))),
            ]
        ],
    )
    table_rows = [
        dmc.TableTr(
            [
                dmc.TableTd(
                    dmc.Text(
                        f"{row['forecast_start']} to {row['forecast_end']}",
                        size="sm",
                    )
                ),
                dmc.TableTd(
                    dmc.Text(str(row["train_weeks"]), size="sm", className="mmm-numeric")
                ),
                dmc.TableTd(
                    dmc.Text(fmt_pct(row["mape"]), size="sm", className="mmm-numeric")
                ),
                dmc.TableTd(
                    dmc.Text(
                        fmt_currency(float(row["rmse"])),
                        size="sm",
                        className="mmm-numeric",
                    )
                ),
                dmc.TableTd(
                    dmc.Text(fmt_pct(row["bias"]), size="sm", className="mmm-numeric")
                ),
                dmc.TableTd(
                    dmc.Text(
                        fmt_pct(row["coverage"]),
                        size="sm",
                        className="mmm-numeric",
                    )
                ),
            ]
        )
        for row in rows
    ]
    return dmc.Stack(
        gap="md",
        children=[
            dmc.Text(
                "This panel is a lightweight temporal stress test, not a full Bayesian MMM "
                "backtest. It fits a ridge surrogate on prior weeks and forecasts the next "
                f"{bt['horizon']} weeks. Use it to spot instability, seasonality misses, or "
                "poor short-term extrapolation. A full MMM backtest would refit the Bayesian "
                "model at each cutoff.",
                size="sm",
                c="dimmed",
            ),
            dmc.Alert(
                (
                    f"{signal}: MAPE is {fmt_pct(bt.get('mape'))}, average bias is "
                    f"{fmt_pct(bt.get('bias'))} ({bias_direction}), and the nominal "
                    f"94% surrogate interval covers {fmt_pct(bt.get('coverage'))} "
                    "of holdout observations."
                ),
                color="yellow" if signal.startswith("The surrogate struggles") else "ochre",
                variant="light",
                title="Surrogate forecast check",
            ),
            summary,
            dmc.Table(
                striped=True,
                highlightOnHover=True,
                withTableBorder=False,
                verticalSpacing="sm",
                children=[
                    dmc.TableThead(
                        dmc.TableTr(
                            [
                                dmc.TableTh("Method"),
                                dmc.TableTh("MAPE"),
                                dmc.TableTh("RMSE"),
                                dmc.TableTh("Bias"),
                                dmc.TableTh("Coverage"),
                            ]
                        )
                    ),
                    dmc.TableTbody(benchmark_rows),
                ],
            ),
            dmc.Table(
                striped=True,
                highlightOnHover=True,
                withTableBorder=False,
                verticalSpacing="sm",
                children=[
                    dmc.TableThead(
                        dmc.TableTr(
                            [
                                dmc.TableTh("Forecast window"),
                                dmc.TableTh("Train weeks"),
                                dmc.TableTh("MAPE"),
                                dmc.TableTh("RMSE"),
                                dmc.TableTh("Bias (+ predicted above actual)"),
                                dmc.TableTh("Surrogate coverage"),
                            ]
                        )
                    ),
                    dmc.TableTbody(table_rows),
                ],
            ),
        ],
    )


def build_overview_toolbar(result: ModelResult) -> dmc.Box:
    """Mounted once in the app shell so preset IDs always exist for callbacks."""
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
                            color="ochre",
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
    )

    return dmc.Box(
        id=OVERVIEW_TOOLBAR,
        style={**OVERVIEW_TOOLBAR_STYLE, "display": "block"},
        children=[range_controls],
    )


def build_overview(result: ModelResult) -> dmc.Stack:
    dmin, dmax = _bounds(result)

    charts = dmc.Grid(
        gutter="md",
        children=[
            dmc.GridCol(
                section(
                    "Actual vs predicted revenue",
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
                    "Revenue decomposition",
                    (
                        "Intercept, trend, seasonality, controls, and channel lift for the selected window."
                        if TREND_COLUMNS
                        else "Intercept, seasonality, controls, and channel lift for the selected window."
                    ),
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
        id="overview-diagnostics-accordion",
        multiple=True,
        variant="separated",
        radius="md",
        chevronPosition="right",
        value=["backtest"],
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
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        dmc.Stack(
                            gap=2,
                            children=[
                                dmc.Text("Fast holdout check", size="sm", fw=600),
                                dmc.Text(
                                    "Surrogate forecast check on unseen weeks.",
                                    size="xs",
                                    c="dimmed",
                                ),
                            ],
                        )
                    ),
                    dmc.AccordionPanel(backtest_panel(result)),
                ],
                value="backtest",
            ),
        ],
    )

    return dmc.Stack(
        gap="lg",
        children=[
            page_header(
                "Model overview",
                f"{result.geo} · weekly media-mix model with {len(result.channels)} "
                "paid channels.",
            ),
            dmc.Box(
                id=OVERVIEW_KPI_GRID_ID,
                children=_build_kpi_grid(result, result, dmin, dmax),
            ),
            charts,
            diagnostics,
        ],
    )

"""Budget Optimiser page: weight sliders with a locked total budget."""

from __future__ import annotations

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, dcc
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from components import CHANNEL_COLORS, apply_dark_theme, kpi_card, page_header, section
from components.ids import MODEL_REFRESH_STORE
from model.mmm import ModelResult, optimise_budget, recommended_weekly_allocation

SLIDER_ID = {"type": "budget-slider", "channel": ""}
WEIGHT_LABEL_ID = {"type": "weight-label", "channel": ""}
ALLOC_LABEL_ID = {"type": "alloc-label", "channel": ""}
RESET_BUTTON_ID = "budget-reset"
APPLY_MODEL_MIX_ID = "budget-apply-model-mix"
PREDICTED_REV_ID = "budget-predicted-revenue"
UPLIFT_ID = "budget-predicted-uplift"
UTILISATION_ID = "budget-utilisation"
ROI_TABLE_ID = "budget-roi-table"
ALLOC_CURRENT_GRAPH_ID = "budget-alloc-current-pie"
ALLOC_PROPOSED_GRAPH_ID = "budget-alloc-proposed-pie"
ALLOC_RECOMMENDED_GRAPH_ID = "budget-alloc-recommended-pie"


def _fmt_currency(v: float) -> str:
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:.1f}K"
    return f"${v:,.0f}"


def _current_weekly_alloc(result: ModelResult) -> dict[str, float]:
    return {c: float(result.spend[c].mean()) for c in result.channels}


def _allocation_donut(
    channels: list[str], alloc: dict[str, float], *, title: str
) -> go.Figure:
    values = [max(0.0, float(alloc.get(c, 0.0))) for c in channels]
    total = sum(values) or 1.0
    fig = go.Figure(
        data=[
            go.Pie(
                labels=channels,
                values=values,
                hole=0.55,
                marker=dict(colors=[CHANNEL_COLORS[c] for c in channels]),
                textinfo="label+percent",
                textposition="outside",
                hovertemplate="%{label}<br>%{percent}<br>$%{value:,.0f} / week<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5, xanchor="center"),
        showlegend=False,
        margin=dict(t=44, b=16, l=16, r=16),
    )
    return apply_dark_theme(fig, height=300)


def _weights_from_weekly_alloc(
    channels: list[str], alloc: dict[str, float]
) -> dict[str, int]:
    total = sum(alloc.get(c, 0.0) for c in channels) or 1.0
    weights = {c: round(float(alloc[c]) / total * 100) for c in channels}
    drift = 100 - sum(weights.values())
    if drift and channels:
        weights[channels[0]] += drift
    return weights


def _default_weights(result: ModelResult) -> dict[str, int]:
    alloc = _current_weekly_alloc(result)
    return _weights_from_weekly_alloc(result.channels, alloc)


def _channel_row(channel: str, weight: int, weekly_amount: float) -> dmc.Stack:
    return dmc.Stack(
        gap=6,
        children=[
            dmc.Group(
                justify="space-between",
                align="center",
                children=[
                    dmc.Group(
                        gap="sm",
                        children=[
                            dmc.Box(
                                w=10,
                                h=10,
                                bg=CHANNEL_COLORS[channel],
                                style={"borderRadius": "2px"},
                            ),
                            dmc.Text(channel, fw=600, size="sm"),
                        ],
                    ),
                    dmc.Group(
                        gap="md",
                        children=[
                            dmc.Text(
                                f"{weight}% weight",
                                size="xs",
                                c="dimmed",
                                id={**WEIGHT_LABEL_ID, "channel": channel},
                                className="mmm-numeric",
                            ),
                            dmc.Text(
                                _fmt_currency(weekly_amount) + " / week",
                                size="sm",
                                fw=600,
                                id={**ALLOC_LABEL_ID, "channel": channel},
                                className="mmm-numeric",
                            ),
                        ],
                    ),
                ],
            ),
            dmc.Slider(
                id={**SLIDER_ID, "channel": channel},
                min=0,
                max=100,
                step=1,
                value=weight,
                color="teal",
                marks=[{"value": v, "label": ""} for v in (0, 25, 50, 75, 100)],
                size="md",
                radius="xl",
            ),
        ],
    )


def _roi_rows(
    channels: list[str],
    current: dict[str, float],
    optimised_alloc: dict[str, float],
    current_pred: dict[str, float | dict[str, float]],
    opt_pred: dict[str, float | dict[str, float]],
    weeks: int,
) -> list[dmc.TableTr]:
    rows: list[dmc.TableTr] = []
    cur_contrib = current_pred["channel_contribution"]  # type: ignore[index]
    opt_contrib = opt_pred["channel_contribution"]  # type: ignore[index]
    for c in channels:
        cur_spend_total = current[c] * weeks
        opt_spend_total = optimised_alloc[c] * weeks
        cur_rev = float(cur_contrib[c])
        opt_rev = float(opt_contrib[c])
        cur_roi = cur_rev / cur_spend_total if cur_spend_total > 0 else 0.0
        opt_roi = opt_rev / opt_spend_total if opt_spend_total > 0 else 0.0
        delta_rev = opt_rev - cur_rev
        delta_color = "teal" if delta_rev >= 0 else "red"
        delta_icon = "tabler:trending-up" if delta_rev >= 0 else "tabler:trending-down"
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
                    dmc.TableTd(dmc.Text(_fmt_currency(cur_spend_total), size="sm", className="mmm-numeric")),
                    dmc.TableTd(dmc.Text(_fmt_currency(opt_spend_total), size="sm", className="mmm-numeric")),
                    dmc.TableTd(dmc.Text(f"{cur_roi:.2f}x", size="sm", className="mmm-numeric")),
                    dmc.TableTd(
                        dmc.Text(
                            f"{opt_roi:.2f}x",
                            size="sm",
                            fw=600,
                            c="teal" if opt_roi >= cur_roi else "orange",
                            className="mmm-numeric",
                        )
                    ),
                    dmc.TableTd(
                        dmc.Group(
                            gap=6,
                            children=[
                                DashIconify(
                                    icon=delta_icon,
                                    width=14,
                                    color="#14b8a6" if delta_rev >= 0 else "#ef4444",
                                ),
                                dmc.Text(
                                    ("+" if delta_rev >= 0 else "") + _fmt_currency(delta_rev),
                                    size="sm",
                                    fw=600,
                                    c=delta_color,
                                    className="mmm-numeric",
                                ),
                            ],
                        )
                    ),
                ]
            )
        )
    return rows


def _roi_table(rows: list[dmc.TableTr]) -> dmc.Table:
    return dmc.Table(
        striped=True,
        highlightOnHover=True,
        withTableBorder=False,
        verticalSpacing="sm",
        children=[
            dmc.TableThead(
                dmc.TableTr(
                    [
                        dmc.TableTh("Channel"),
                        dmc.TableTh("Current spend"),
                        dmc.TableTh("New spend"),
                        dmc.TableTh("Current ROI"),
                        dmc.TableTh("New ROI"),
                        dmc.TableTh("Revenue Δ"),
                    ]
                )
            ),
            dmc.TableTbody(rows),
        ],
    )


# ---------- layout ---------------------------------------------------------


def build_optimiser(result: ModelResult) -> dmc.Stack:
    current_alloc = _current_weekly_alloc(result)
    total_weekly = sum(current_alloc.values())
    total_window = total_weekly * result.n_weeks
    weights = _default_weights(result)
    sliders = [_channel_row(c, weights[c], current_alloc[c]) for c in result.channels]

    current_pred = optimise_budget(result, current_alloc)
    roi_rows = _roi_rows(
        result.channels, current_alloc, current_alloc, current_pred, current_pred, result.n_weeks
    )

    total_rev = float(current_pred["total_revenue"])

    predicted_kpi = dmc.Paper(
        p="lg",
        radius="md",
        shadow="sm",
        withBorder=False,
        className="mmm-paper",
        children=dmc.Group(
            justify="space-between",
            align="flex-start",
            wrap="nowrap",
            children=[
                dmc.Stack(
                    gap=6,
                    children=[
                        dmc.Text(
                            "Predicted revenue",
                            size="xs",
                            c="dimmed",
                            tt="uppercase",
                            fw=600,
                        ),
                        dmc.Text(
                            _fmt_currency(total_rev),
                            size="xl",
                            fw=700,
                            id=PREDICTED_REV_ID,
                            className="mmm-numeric",
                        ),
                        dmc.Text(
                            "Baseline plus steady-state paid media contribution",
                            size="xs",
                            c="dimmed",
                        ),
                    ],
                ),
                dmc.ThemeIcon(
                    DashIconify(icon="tabler:cash", width=20),
                    variant="light",
                    color="teal",
                    size="lg",
                    radius="md",
                ),
            ],
        ),
    )
    uplift_kpi = dmc.Paper(
        p="lg",
        radius="md",
        shadow="sm",
        withBorder=False,
        className="mmm-paper",
        children=dmc.Group(
            justify="space-between",
            align="flex-start",
            wrap="nowrap",
            children=[
                dmc.Stack(
                    gap=6,
                    children=[
                        dmc.Text(
                            "Uplift vs current",
                            size="xs",
                            c="dimmed",
                            tt="uppercase",
                            fw=600,
                        ),
                        dmc.Text(
                            "+$0",
                            size="xl",
                            fw=700,
                            id=UPLIFT_ID,
                            className="mmm-numeric",
                        ),
                        dmc.Text(
                            "Move the sliders to reallocate",
                            size="xs",
                            c="dimmed",
                            id=UTILISATION_ID,
                        ),
                    ],
                ),
                dmc.ThemeIcon(
                    DashIconify(icon="tabler:trending-up", width=20),
                    variant="light",
                    color="teal",
                    size="lg",
                    radius="md",
                ),
            ],
        ),
    )

    kpis_row = dmc.SimpleGrid(
        cols={"base": 1, "sm": 3},
        spacing="md",
        children=[
            kpi_card(
                label="Total weekly budget",
                value=_fmt_currency(total_weekly),
                icon="tabler:wallet",
                helper=(
                    f"Locked · {_fmt_currency(total_window)} over "
                    f"{result.n_weeks}-week fit window"
                ),
            ),
            predicted_kpi,
            uplift_kpi,
        ],
    )

    controls = section(
        "Reallocate Budget",
        "Sliders are weights. Total spend is locked; the mix changes proportionally.",
        dmc.Stack(
            gap="lg",
            children=[
                *sliders,
                dmc.Group(
                    justify="space-between",
                    children=[
                        dmc.Text(
                            "Channel weights are normalised to 100% before prediction.",
                            size="xs",
                            c="dimmed",
                        ),
                        dmc.Group(
                            gap="xs",
                            children=[
                                dmc.Button(
                                    "Apply model-suggested mix",
                                    id=APPLY_MODEL_MIX_ID,
                                    leftSection=DashIconify(
                                        icon="tabler:sparkles", width=14
                                    ),
                                    variant="subtle",
                                    color="teal",
                                    size="xs",
                                ),
                                dmc.Button(
                                    "Reset to current mix",
                                    id=RESET_BUTTON_ID,
                                    leftSection=DashIconify(
                                        icon="tabler:refresh", width=14
                                    ),
                                    variant="subtle",
                                    color="gray",
                                    size="xs",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    )

    recommended_alloc = recommended_weekly_allocation(result)
    rec_pred = optimise_budget(result, recommended_alloc)
    rec_rev = float(rec_pred["total_revenue"])

    alloc_charts = section(
        "Allocation mix",
        "Current weekly mix (from data), your proposed mix (sliders), and a model-suggested mix "
        f"(steady-state SLSQP maximising predicted revenue — {_fmt_currency(rec_rev)} at suggested mix).",
        dmc.Grid(
            gutter="md",
            children=[
                dmc.GridCol(
                    dcc.Graph(
                        id=ALLOC_CURRENT_GRAPH_ID,
                        figure=_allocation_donut(
                            result.channels,
                            current_alloc,
                            title="Current (data)",
                        ),
                        config={"displayModeBar": False},
                    ),
                    span={"base": 12, "md": 4},
                ),
                dmc.GridCol(
                    dcc.Graph(
                        id=ALLOC_PROPOSED_GRAPH_ID,
                        figure=_allocation_donut(
                            result.channels,
                            current_alloc,
                            title="Proposed (sliders)",
                        ),
                        config={"displayModeBar": False},
                    ),
                    span={"base": 12, "md": 4},
                ),
                dmc.GridCol(
                    dcc.Graph(
                        id=ALLOC_RECOMMENDED_GRAPH_ID,
                        figure=_allocation_donut(
                            result.channels,
                            recommended_alloc,
                            title="Model-suggested",
                        ),
                        config={"displayModeBar": False},
                    ),
                    span={"base": 12, "md": 4},
                ),
            ],
        ),
    )

    table = section(
        "Current vs Optimised",
        "Per-channel spend, ROI, and revenue delta at the proposed mix.",
        dmc.Box(id=ROI_TABLE_ID, children=_roi_table(roi_rows)),
    )

    title_block = page_header(
        "Budget Optimiser",
        "Explore reallocations at a fixed total budget and see the predicted revenue response.",
    )

    return dmc.Stack(
        gap="lg",
        children=[
            title_block,
            kpis_row,
            controls,
            alloc_charts,
            table,
        ],
    )


# ---------- callbacks ------------------------------------------------------


def register_optimiser_callbacks(app, results_by_geo: dict[str, ModelResult]) -> None:
    @app.callback(
        Output(PREDICTED_REV_ID, "children"),
        Output(UPLIFT_ID, "children"),
        Output(UPLIFT_ID, "c"),
        Output(UTILISATION_ID, "children"),
        Output(ROI_TABLE_ID, "children"),
        Output(ALLOC_PROPOSED_GRAPH_ID, "figure"),
        Output({**WEIGHT_LABEL_ID, "channel": ALL}, "children"),
        Output({**ALLOC_LABEL_ID, "channel": ALL}, "children"),
        Input({**SLIDER_ID, "channel": ALL}, "value"),
        Input(MODEL_REFRESH_STORE, "data"),
        State({**SLIDER_ID, "channel": ALL}, "id"),
    )
    def _recalc(values, _refresh, ids):
        result = results_by_geo["All"]
        channels = [i["channel"] for i in ids]

        raw = [float(v or 0) for v in values]
        total_w = sum(raw) or 1.0
        weights_pct = [v / total_w for v in raw]

        current_alloc = _current_weekly_alloc(result)
        total_weekly = sum(current_alloc.values())

        new_alloc = {c: weights_pct[i] * total_weekly for i, c in enumerate(channels)}

        current_pred = optimise_budget(result, current_alloc)
        new_pred = optimise_budget(result, new_alloc)

        cur_rev = float(current_pred["total_revenue"])
        new_rev = float(new_pred["total_revenue"])
        uplift = new_rev - cur_rev
        uplift_color = "teal" if uplift >= 0 else "red"
        uplift_str = ("+" if uplift >= 0 else "") + _fmt_currency(uplift)
        pct_change = (uplift / cur_rev * 100) if cur_rev else 0.0
        utilisation = f"{pct_change:+.2f}% vs current allocation"

        rows = _roi_rows(
            channels, current_alloc, new_alloc, current_pred, new_pred, result.n_weeks
        )

        prop_fig = _allocation_donut(
            channels, new_alloc, title="Proposed (sliders)"
        )

        weight_children = [f"{round(w * 100)}% weight" for w in weights_pct]
        alloc_children = [_fmt_currency(new_alloc[c]) + " / week" for c in channels]

        return (
            _fmt_currency(new_rev),
            uplift_str,
            uplift_color,
            utilisation,
            _roi_table(rows),
            prop_fig,
            weight_children,
            alloc_children,
        )

    @app.callback(
        Output({**SLIDER_ID, "channel": ALL}, "value"),
        Input(RESET_BUTTON_ID, "n_clicks"),
        Input(APPLY_MODEL_MIX_ID, "n_clicks"),
        State({**SLIDER_ID, "channel": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _preset_sliders(_reset_clicks, _apply_clicks, ids):
        from dash import ctx

        if not ctx.triggered_id:
            raise PreventUpdate

        result = results_by_geo["All"]
        if ctx.triggered_id == APPLY_MODEL_MIX_ID:
            rec = recommended_weekly_allocation(result)
            weights = _weights_from_weekly_alloc(result.channels, rec)
        else:
            weights = _default_weights(result)
        return [weights[i["channel"]] for i in ids]

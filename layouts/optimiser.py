"""Budget optimiser layout and shared presentation helpers."""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import dcc
from dash_iconify import DashIconify

from components import CHANNEL_COLORS, kpi_card, page_header, section
from figures.optimiser import _allocation_donut
from model.mmm import (
    ModelResult,
    current_weekly_allocation,
    observed_spend_support,
    optimise_budget,
    posterior_budget_summary,
    recommended_weekly_allocation,
    steady_state_current_budget_prediction,
)

SLIDER_ID = {"type": "budget-slider", "channel": ""}
WEIGHT_LABEL_ID = {"type": "weight-label", "channel": ""}
ALLOC_LABEL_ID = {"type": "alloc-label", "channel": ""}
RESET_BUTTON_ID = "budget-reset"
APPLY_MODEL_MIX_ID = "budget-apply-model-mix"
CONSTRAINT_MIN_ID = {"type": "budget-min-spend", "channel": ""}
CONSTRAINT_MAX_ID = {"type": "budget-max-spend", "channel": ""}
OPTIMISER_STATUS_ID = "budget-optimiser-status"
PREDICTED_REV_ID = "budget-predicted-revenue"
PREDICTED_HDI_ID = "budget-predicted-hdi"
UPLIFT_ID = "budget-predicted-uplift"
UPLIFT_PROB_ID = "budget-uplift-probability"
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


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _current_weekly_alloc(result: ModelResult) -> dict[str, float]:
    return current_weekly_allocation(result)


def _support_status(result: ModelResult, channel: str, weekly_spend: float) -> tuple[str, str]:
    p5, p95 = observed_spend_support(result, channel)
    if weekly_spend < p5:
        return f"Below P5 ({_fmt_currency(p5)}/wk)", "orange"
    if weekly_spend > p95:
        return f"Above P95 ({_fmt_currency(p95)}/wk)", "orange"
    return "Inside P5-P95", "dimmed"


def _outside_support_count(result: ModelResult, alloc: dict[str, float]) -> int:
    count = 0
    for c in result.channels:
        p5, p95 = observed_spend_support(result, c)
        w = float(alloc.get(c, 0.0))
        if w < p5 or w > p95:
            count += 1
    return count


def _weights_from_weekly_alloc(
    channels: list[str], alloc: dict[str, float]
) -> dict[str, float]:
    total = sum(alloc.get(c, 0.0) for c in channels) or 1.0
    return {c: float(alloc[c]) / total * 100.0 for c in channels}


def _default_weights(result: ModelResult) -> dict[str, float]:
    alloc = _current_weekly_alloc(result)
    return _weights_from_weekly_alloc(result.channels, alloc)


def _fmt_weight(weight: float) -> str:
    return f"{weight:.1f}% weight"


def _constraint_dict(
    channels: list[str],
    values: list[float | int | None] | None,
) -> dict[str, float]:
    """Map optional NumberInput values to per-channel weekly spend constraints."""
    if not values:
        return {}
    out: dict[str, float] = {}
    for i, channel in enumerate(channels):
        if i >= len(values):
            break
        value = values[i]
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric >= 0:
            out[channel] = numeric
    return out


def _channel_row(channel: str, weight: float, weekly_amount: float) -> dmc.Stack:
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
                                _fmt_weight(weight),
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
                step=0.1,
                value=weight,
                color="teal",
                marks=[{"value": v, "label": ""} for v in (0, 25, 50, 75, 100)],
                size="md",
                radius="xl",
            ),
        ],
    )


def _constraint_row(channel: str, current_weekly: float) -> dmc.TableTr:
    return dmc.TableTr(
        [
            dmc.TableTd(
                dmc.Group(
                    gap="sm",
                    children=[
                        dmc.Box(
                            w=10,
                            h=10,
                            bg=CHANNEL_COLORS[channel],
                            style={"borderRadius": "2px"},
                        ),
                        dmc.Text(channel, fw=500, size="sm"),
                    ],
                )
            ),
            dmc.TableTd(
                dmc.Text(
                    _fmt_currency(current_weekly) + " / week",
                    size="sm",
                    className="mmm-numeric",
                )
            ),
            dmc.TableTd(
                dmc.NumberInput(
                    id={**CONSTRAINT_MIN_ID, "channel": channel},
                    value=None,
                    min=0,
                    step=1000,
                    placeholder="No min",
                    size="xs",
                )
            ),
            dmc.TableTd(
                dmc.NumberInput(
                    id={**CONSTRAINT_MAX_ID, "channel": channel},
                    value=None,
                    min=0,
                    step=1000,
                    placeholder="No max",
                    size="xs",
                )
            ),
        ]
    )


def _constraints_table(result: ModelResult) -> dmc.Table:
    current_alloc = _current_weekly_alloc(result)
    return dmc.Table(
        striped=True,
        highlightOnHover=True,
        withTableBorder=False,
        verticalSpacing="xs",
        children=[
            dmc.TableThead(
                dmc.TableTr(
                    [
                        dmc.TableTh("Channel"),
                        dmc.TableTh("Current"),
                        dmc.TableTh("Min weekly spend"),
                        dmc.TableTh("Max weekly spend"),
                    ]
                )
            ),
            dmc.TableTbody(
                [_constraint_row(c, current_alloc[c]) for c in result.channels]
            ),
        ],
    )


def _roi_rows(
    result: ModelResult,
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
        support_label, support_color = _support_status(result, c, optimised_alloc[c])
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
                    dmc.TableTd(dmc.Text(support_label, size="sm", c=support_color)),
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
                        dmc.TableTh("Support"),
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

    current_pred = steady_state_current_budget_prediction(result)
    current_summary = posterior_budget_summary(
        result,
        current_alloc,
        current_allocation=current_alloc,
    )
    roi_rows = _roi_rows(
        result,
        result.channels,
        current_alloc,
        current_alloc,
        current_pred,
        current_pred,
        result.n_weeks,
    )

    total_rev = float(current_summary["expected_revenue"])
    hdi_low, hdi_high = current_summary["revenue_hdi"]  # type: ignore[assignment]

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
                            f"94% HDI {_fmt_currency(float(hdi_low))}–{_fmt_currency(float(hdi_high))}",
                            size="xs",
                            c="dimmed",
                            id=PREDICTED_HDI_ID,
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
                        dmc.Text(
                            "P(proposed > current): 0%",
                            size="xs",
                            c="dimmed",
                            id=UPLIFT_PROB_ID,
                            className="mmm-numeric",
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

    constraints = section(
        "Optimisation constraints",
        "Optional weekly min/max spend bounds used when re-running the model-suggested mix.",
        dmc.Stack(
            gap="sm",
            children=[
                _constraints_table(result),
                dmc.Group(
                    justify="space-between",
                    align="center",
                    children=[
                        dmc.Box(id=OPTIMISER_STATUS_ID),
                        dmc.Button(
                            "Re-run optimisation",
                            id=APPLY_MODEL_MIX_ID,
                            leftSection=DashIconify(
                                icon="tabler:sparkles", width=14
                            ),
                            color="teal",
                            variant="light",
                            size="xs",
                        ),
                    ],
                ),
            ],
        ),
    )

    recommended_alloc = recommended_weekly_allocation(result)
    rec_pred = optimise_budget(result, recommended_alloc)
    rec_rev = float(rec_pred["total_revenue"])
    rec_outside = _outside_support_count(result, recommended_alloc)

    alloc_charts = section(
        "Allocation mix",
        "Current weekly mix (from data), your proposed mix (sliders), and a model-suggested "
        f"steady-state mix ({_fmt_currency(rec_rev)} at suggested mix; "
        f"{rec_outside} channel{'s' if rec_outside != 1 else ''} outside observed P5-P95 support).",
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
        "Current vs proposed",
        "Current and proposed scenarios both use steady-state response curves.",
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
            constraints,
            alloc_charts,
            table,
        ],
    )

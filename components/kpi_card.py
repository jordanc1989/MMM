"""Reusable KPI card component."""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify


def kpi_card(
    *,
    label: str,
    value: str,
    icon: str,
    helper: str | None = None,
    sub: str | None = None,
    yoy: str | None = None,
    yoy_color: str = "dimmed",
    accent: str = "teal",
) -> dmc.Paper:
    """A shadow-elevated paper showing a labelled metric with a Tabler icon.

    Args:
        label: uppercase caption displayed above the value.
        value: the primary numeric or text value.
        icon: Tabler icon slug (e.g. `tabler:chart-bar`).
        helper: optional secondary line rendered under the value.
        sub: optional tertiary dimmed line (used e.g. for HDI bands).
        yoy: optional year-over-year line (e.g. ``YoY +4.2%``).
        yoy_color: Mantine color name for ``yoy`` (e.g. ``teal``, ``red``, ``dimmed``).
        accent: Mantine color used for the ThemeIcon.
    """
    body: list = [
        dmc.Text(
            label,
            size="xs",
            c="dimmed",
            tt="uppercase",
            fw=600,
        ),
        dmc.Text(
            value,
            size="xl",
            fw=700,
            className="mmm-numeric",
        ),
    ]
    if helper:
        body.append(dmc.Text(helper, size="xs", c="dimmed"))
    if sub:
        body.append(dmc.Text(sub, size="xs", c="dimmed", fs="italic"))
    if yoy:
        body.append(
            dmc.Text(yoy, size="xs", fw=600, c=yoy_color, className="mmm-numeric")
        )

    return dmc.Paper(
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
                dmc.Stack(children=body, gap=6),
                dmc.ThemeIcon(
                    DashIconify(icon=icon, width=20),
                    variant="light",
                    color=accent,
                    size="lg",
                    radius="md",
                ),
            ],
        ),
    )

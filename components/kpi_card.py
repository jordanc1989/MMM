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
    accent: str = "teal",
) -> dmc.Paper:
    """A shadow-elevated paper showing a labelled metric with a Tabler icon.

    Args:
        label: uppercase caption displayed above the value.
        value: the primary numeric or text value.
        icon: Tabler icon slug (e.g. `tabler:chart-bar`).
        helper: optional secondary line rendered under the value.
        sub: optional tertiary dimmed line (used e.g. for HDI bands).
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

"""Reusable KPI card component (flat, editorial)."""

from __future__ import annotations

import dash_mantine_components as dmc


def kpi_card(
    *,
    label: str,
    value: str,
    icon: str | None = None,
    helper: str | None = None,
    sub: str | None = None,
    yoy: str | None = None,
    yoy_color: str = "dimmed",
    accent: str = "ochre",
) -> dmc.Paper:
    """A flat, hairline-bordered metric panel: small-caps label over a mono value.

    Args:
        label: small-caps caption displayed above the value.
        value: the primary numeric or text value (rendered in the mono face).
        icon: accepted for call-site compatibility; no longer rendered.
        helper: optional secondary line rendered under the value.
        sub: optional tertiary dimmed line (used e.g. for HDI bands).
        yoy: optional year-over-year line (e.g. ``YoY +4.2%``).
        yoy_color: Mantine colour name or hex for ``yoy``.
        accent: accepted for call-site compatibility; no longer rendered.
    """
    del icon, accent  # retained in the signature for call-site compatibility

    body: list = [
        dmc.Text(label, className="mmm-eyebrow"),
        dmc.Text(value, fz=28, fw=600, lh=1.1, className="mmm-numeric"),
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
        radius="sm",
        withBorder=False,
        className="mmm-paper",
        children=dmc.Stack(children=body, gap=8),
    )

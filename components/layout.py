"""Page-level layout helpers."""

from __future__ import annotations

from typing import Any

import dash_mantine_components as dmc


def page_header(title: str, description: str) -> dmc.Stack:
    """Left-aligned serif page title + supporting description."""
    return dmc.Stack(
        gap=4,
        children=[
            dmc.Title(title, order=2, fw=600),
            dmc.Text(description, size="sm", c="dimmed", maw=760),
        ],
    )


def section(title: str, description: str | None, children: Any) -> dmc.Paper:
    """A flat, hairline-bordered panel with a ruled serif header strip."""
    header_text: list = [dmc.Text(title, fw=600, fz="md", className="mmm-serif")]
    if description:
        header_text.append(dmc.Text(description, size="xs", c="dimmed"))

    return dmc.Paper(
        p="lg",
        radius="sm",
        withBorder=False,
        className="mmm-paper",
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Box(
                    pb="sm",
                    className="mmm-rule",
                    children=dmc.Stack(gap=2, children=header_text),
                ),
                children,
            ],
        ),
    )

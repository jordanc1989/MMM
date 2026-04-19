"""Page-level layout helpers."""

from __future__ import annotations

from typing import Any

import dash_mantine_components as dmc


def page_header(title: str, description: str) -> dmc.Stack:
    """Left-aligned page title + supporting description."""
    return dmc.Stack(
        gap=4,
        children=[
            dmc.Title(title, order=2, fw=600),
            dmc.Text(description, size="sm", c="dimmed"),
        ],
    )


def section(title: str, description: str | None, children: Any) -> dmc.Paper:
    """A bordered-by-shadow content block with a small header strip."""
    header: list = [dmc.Text(title, size="sm", fw=600)]
    if description:
        header.append(dmc.Text(description, size="xs", c="dimmed"))

    return dmc.Paper(
        p="lg",
        radius="md",
        shadow="sm",
        withBorder=False,
        className="mmm-paper",
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Stack(gap=2, children=header),
                children,
            ],
        ),
    )

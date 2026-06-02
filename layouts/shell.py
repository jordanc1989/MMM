"""Application shell, navigation, and refit overlay layout."""

from __future__ import annotations

import dash_mantine_components as dmc
import dash
from dash import dcc, html
from dash_iconify import DashIconify

from components.ids import (
    MODEL_REFRESH_STORE,
    OPT_DRAWS,
    OPT_REFIT_BTN,
    OPT_REFIT_STATUS,
    OPT_TARGET_ACCEPT,
    OPT_TUNE,
    REFIT_JOB_STORE,
    REFIT_OVERLAY_ROOT,
    REFIT_OVERLAY_STORE,
    REFIT_POLL_INTERVAL,
    REFIT_PROGRESS_CHAINS,
)
from content.methodology import REFIT_OVERLAY_DESCRIPTION, REFIT_OVERLAY_TITLE
from dashboard.data import overview_date_store
from dashboard.theme import (
    ACCENT,
    APP_MIN_HEIGHT_STYLE,
    PAGE_CONTAINER_STYLE,
    REFIT_OVERLAY_STYLE,
    mantine_theme,
)
from layouts.overview import build_overview_toolbar
from model.mmm import ModelResult, load_sampler_config


def nav_pages() -> list[dict]:
    return [
        page
        for page in dash.page_registry.values()
        if page.get("path") and not page.get("hide_from_nav")
    ]


def _nav_link(page: dict) -> dmc.NavLink:
    path = str(page["path"])
    return dmc.NavLink(
        id={"type": "nav", "path": path},
        label=page["name"],
        leftSection=DashIconify(
            icon=page.get("icon", "tabler:circle"),
            width=16,
        ),
        href=page.get("relative_path", path),
        active=False,
        variant="subtle",
        color="gray",
    )


def header(result: ModelResult) -> dmc.AppShellHeader:
    defaults = load_sampler_config()
    sc = result.sampler_config or {}
    ta = float(sc.get("target_accept", defaults["target_accept"]))
    draws = int(sc.get("draws", defaults["draws"]))
    tune = int(sc.get("tune", defaults["tune"]))

    return dmc.AppShellHeader(
        px="lg",
        children=dmc.Group(
            h="100%",
            justify="space-between",
            align="center",
            children=[
                dmc.Stack(
                    gap=0,
                    children=[
                        dmc.Text(
                            "Bayesian Media Mix Model",
                            fw=600,
                            fz="lg",
                            className="mmm-serif",
                        ),
                        dmc.Text(
                            f"{result.geo} · weekly simulated Meridian data",
                            size="xs",
                            c="dimmed",
                        ),
                    ],
                ),
                dmc.Group(
                    gap="md",
                    align="center",
                    children=[
                        dmc.Popover(
                            width=320,
                            position="bottom-end",
                            shadow="md",
                            withArrow=True,
                            children=[
                                dmc.PopoverTarget(
                                    dmc.Button(
                                        "Options",
                                        size="xs",
                                        variant="light",
                                        color="gray",
                                        leftSection=DashIconify(
                                            icon="tabler:adjustments", width=14
                                        ),
                                    )
                                ),
                                dmc.PopoverDropdown(
                                    p="md",
                                    children=dmc.Stack(
                                        gap="sm",
                                        children=[
                                            dmc.Text(
                                                "NUTS sampling",
                                                size="xs",
                                                tt="uppercase",
                                                fw=600,
                                                c="dimmed",
                                            ),
                                            dmc.NumberInput(
                                                id=OPT_TARGET_ACCEPT,
                                                label="Target accept",
                                                description="0.75-0.999",
                                                value=ta,
                                                min=0.75,
                                                max=0.9999,
                                                step=0.005,
                                                decimalScale=4,
                                                size="sm",
                                            ),
                                            dmc.NumberInput(
                                                id=OPT_DRAWS,
                                                label="Draws",
                                                value=draws,
                                                min=100,
                                                max=20000,
                                                step=100,
                                                size="sm",
                                            ),
                                            dmc.NumberInput(
                                                id=OPT_TUNE,
                                                label="Tune (warmup)",
                                                value=tune,
                                                min=200,
                                                max=50000,
                                                step=100,
                                                size="sm",
                                            ),
                                            dmc.Button(
                                                "Apply & refit",
                                                id=OPT_REFIT_BTN,
                                                color="ochre",
                                                variant="filled",
                                                size="sm",
                                                fullWidth=True,
                                                n_clicks=0,
                                                loading=False,
                                            ),
                                            dmc.Box(id=OPT_REFIT_STATUS),
                                        ],
                                    ),
                                ),
                            ],
                        ),
                        dmc.Text(
                            f"{result.geo} · demo",
                            size="xs",
                            c="dimmed",
                            className="mmm-numeric",
                        ),
                    ],
                ),
            ],
        ),
    )


def navbar(result: ModelResult) -> dmc.AppShellNavbar:
    return dmc.AppShellNavbar(
        p="md",
        children=dmc.Stack(
            gap="xs",
            children=[
                *[_nav_link(page) for page in nav_pages()],
                dmc.Divider(my="md"),
                dmc.Text("Model", size="xs", c="dimmed", tt="uppercase", fw=600),
                dmc.Stack(
                    gap=2,
                    children=[
                        dmc.Text(result.geo, size="sm", fw=600),
                        dmc.Text(
                            f"{len(result.channels)} paid channels",
                            size="xs",
                            c="dimmed",
                        ),
                        dmc.Text(
                            "Geometric adstock, logistic saturation",
                            size="xs",
                            c="dimmed",
                        ),
                    ],
                ),
            ],
        ),
    )


def refit_progress_placeholder() -> dmc.Stack:
    return dmc.Stack(
        gap=6,
        children=[
            dmc.Text("Starting sampler...", size="xs", c="dimmed"),
            dmc.Progress(
                value=None,
                striped=True,
                animated=True,
                color="ochre",
                size="sm",
                radius="xl",
            ),
        ],
    )


def refit_progress_from_snapshot(snapshot: dict | None) -> dmc.Stack:
    """Render phase-level progress from ``SamplingProgressTracker.snapshot()``."""
    if not snapshot or not snapshot.get("chains"):
        return refit_progress_placeholder()

    phase = snapshot.get("phase") or "idle"
    if phase == "nuts" and snapshot.get("nuts_indeterminate"):
        return dmc.Stack(
            gap=6,
            children=[
                dmc.Text("NUTS sampling (nutpie / JAX)...", size="xs", c=ACCENT),
                dmc.Progress(
                    value=None,
                    striped=True,
                    animated=True,
                    color="ochre",
                    size="sm",
                    radius="xl",
                ),
                dmc.Text(
                    "Detailed sampler progress is available in the terminal.",
                    size="xs",
                    c="dimmed",
                ),
            ],
        )

    parts: list = []
    if phase == "ppc":
        parts.extend(
            [
                dmc.Text(
                    "Posterior predictive sampling...",
                    size="xs",
                    c=ACCENT,
                ),
                dmc.Progress(
                    value=None,
                    striped=True,
                    animated=True,
                    color="ochre",
                    size="sm",
                    radius="xl",
                ),
            ]
        )

    for chain in snapshot["chains"]:
        warm = "Warmup" if chain.get("warmup") else "Sampling"
        pct = float(chain.get("pct") or 0.0)
        parts.append(
            dmc.Stack(
                gap=4,
                children=[
                    dmc.Group(
                        justify="space-between",
                        wrap="nowrap",
                        children=[
                            dmc.Text(
                                f"Chain {int(chain['chain']) + 1} - {warm}",
                                size="xs",
                                fw=500,
                            ),
                            dmc.Text(f"{pct:.0f}%", size="xs", c="dimmed"),
                        ],
                    ),
                    dmc.Progress(value=pct, color="ochre", size="sm", radius="xl"),
                ],
            )
        )

    if not snapshot.get("nuts_indeterminate"):
        parts.append(
            dmc.Text(
                f"Mean across chains {float(snapshot.get('overall_pct') or 0):.0f}%.",
                size="xs",
                c="dimmed",
            )
        )
    return dmc.Stack(gap="sm", children=parts)


def refit_overlay_root() -> dmc.Box:
    return dmc.Box(
        id=REFIT_OVERLAY_ROOT,
        style={**REFIT_OVERLAY_STYLE, "display": "none"},
        children=dmc.Paper(
            p="lg",
            maw=460,
            w="100%",
            mx="md",
            radius="md",
            withBorder=True,
            shadow="md",
            className="mmm-paper",
            children=dmc.Stack(
                gap="md",
                children=[
                    dmc.Group(
                        justify="space-between",
                        align="flex-start",
                        wrap="nowrap",
                        children=[
                            dmc.Stack(
                                gap=4,
                                children=[
                                    dmc.Text(REFIT_OVERLAY_TITLE, fw=600, size="md"),
                                    dmc.Text(
                                        REFIT_OVERLAY_DESCRIPTION,
                                        size="sm",
                                        c="dimmed",
                                    ),
                                ],
                            ),
                            dmc.Loader(color="ochre", size="md", type="oval"),
                        ],
                    ),
                    dmc.Stack(
                        gap=6,
                        children=[
                            dmc.Text(
                                "Sampling progress",
                                size="xs",
                                tt="uppercase",
                                fw=600,
                                c="dimmed",
                            ),
                            html.Div(
                                id=REFIT_PROGRESS_CHAINS,
                                children=refit_progress_placeholder(),
                            ),
                        ],
                    ),
                ],
            ),
        ),
    )


def build_shell(result: ModelResult) -> dmc.MantineProvider:
    return dmc.MantineProvider(
        theme=mantine_theme(),
        forceColorScheme="dark",
        children=dmc.Box(
            pos="relative",
            style=APP_MIN_HEIGHT_STYLE,
            children=[
                dcc.Store(id=MODEL_REFRESH_STORE, data=0),
                dcc.Store(id=REFIT_OVERLAY_STORE, data={"open": False}),
                dcc.Store(id=REFIT_JOB_STORE, data={"running": False}),
                dcc.Interval(
                    id=REFIT_POLL_INTERVAL,
                    interval=400,
                    n_intervals=0,
                    max_intervals=-1,
                    disabled=True,
                ),
                dmc.AppShell(
                    header={"height": 64},
                    navbar={"width": 260, "breakpoint": "sm"},
                    padding="lg",
                    children=[
                        dcc.Location(id="url", refresh=False),
                        overview_date_store(result),
                        header(result),
                        navbar(result),
                        dmc.AppShellMain(
                            dmc.Stack(
                                gap=0,
                                children=[
                                    build_overview_toolbar(result),
                                    dmc.Container(
                                        fluid=True,
                                        px=0,
                                        style=PAGE_CONTAINER_STYLE,
                                        children=dash.page_container,
                                    ),
                                ],
                            )
                        ),
                    ],
                ),
                refit_overlay_root(),
            ],
        ),
    )

"""MMM Demo Dashboard - Dash + DMC entrypoint.

Run with: `python app.py` (serves on http://127.0.0.1:8050).
"""

from __future__ import annotations

import dash_mantine_components as dmc
import pandas as pd
from dash import Dash, Input, Output, _dash_renderer, dcc
from dash_iconify import DashIconify

from components.ids import GLOBAL_DATE_RANGE
from data.loader import aggregate_geo, load_meridian
from model.mmm import ModelResult, fit_surrogate
from pages import (
    build_contributions,
    build_optimiser,
    build_overview,
    build_response_curves,
    register_contributions_callbacks,
    register_optimiser_callbacks,
    register_overview_callbacks,
    register_response_curve_callbacks,
)

_dash_renderer._set_react_version("18.2.0")

ROUTES = [
    ("/", "Overview", "tabler:layout-dashboard"),
    ("/contributions", "Contributions", "tabler:chart-area-line"),
    ("/response-curves", "Response Curves", "tabler:chart-ppf"),
    ("/optimiser", "Optimiser", "tabler:adjustments-horizontal"),
]


def _nav_link(path: str, label: str, icon: str) -> dmc.NavLink:
    """Nav item with Tabler icon (same treatment for every route)."""
    glyph = DashIconify(icon=icon, width=16)
    return dmc.NavLink(
        id={"type": "nav", "path": path},
        label=label,
        leftSection=glyph,
        href=path,
        active=False,
        variant="light",
        color="teal",
    )


# ---------- model cache ---------------------------------------------------


def build_model_cache() -> dict[str, ModelResult]:
    """Fit (or load from disk cache) the Bayesian MMM on all geographies aggregated.

    First launch samples with PyMC/NUTS (~60-90s, cached to `data/mmm_idata.nc`).
    Subsequent launches load the cached inference data and are near-instant.
    """
    df = load_meridian()
    print("Loading MMM (first run ~60-90s while PyMC samples)...", flush=True)
    result = fit_surrogate(aggregate_geo(df, None), "All")
    print("MMM ready.", flush=True)
    return {"All": result}


# ---------- shell ---------------------------------------------------------


def _theme() -> dict:
    return {
        "primaryColor": "teal",
        "fontFamily": "DM Sans, sans-serif",
        "headings": {"fontFamily": "DM Sans, sans-serif", "fontWeight": "600"},
        "defaultRadius": "md",
        "colors": {
            "dark": [
                "#d4d6d9",
                "#b1b4ba",
                "#8a8f97",
                "#62666d",
                "#3a3f47",
                "#2a2f36",
                "#23272e",
                "#181c21",
                "#13161a",
                "#0b0d10",
            ],
        },
    }


def _header(result: ModelResult) -> dmc.AppShellHeader:
    dmin = pd.to_datetime(result.dates).min()
    dmax = pd.to_datetime(result.dates).max()
    return dmc.AppShellHeader(
        px="lg",
        children=dmc.Group(
            h="100%",
            justify="space-between",
            align="center",
            children=[
                dmc.Group(
                    gap="sm",
                    children=[
                        dmc.ThemeIcon(
                            DashIconify(icon="tabler:chart-histogram", width=18),
                            variant="light",
                            color="teal",
                            size="lg",
                            radius="md",
                        ),
                        dmc.Stack(
                            gap=0,
                            children=[
                                dmc.Text(
                                    "Meridian Media Mix", fw=700, size="md"
                                ),
                                dmc.Text(
                                    "Bayesian MMM (pymc-marketing) on Meridian simulated data",
                                    size="xs",
                                    c="dimmed",
                                ),
                            ],
                        ),
                    ],
                ),
                dmc.Group(
                    gap="md",
                    align="center",
                    children=[
                        dmc.Stack(
                            gap=4,
                            children=[
                                dmc.Text(
                                    "Analysis period",
                                    size="xs",
                                    c="dimmed",
                                    tt="uppercase",
                                    fw=600,
                                ),
                                dcc.DatePickerRange(
                                    id=GLOBAL_DATE_RANGE,
                                    className="mmm-date-range",
                                    min_date_allowed=dmin,
                                    max_date_allowed=dmax,
                                    start_date=dmin,
                                    end_date=dmax,
                                    display_format="MMM D, YYYY",
                                    start_date_placeholder_text="Start",
                                    end_date_placeholder_text="End",
                                    style={
                                        "backgroundColor": "#181c21",
                                        "color": "#d4d6d9",
                                        "border": "1px solid #3a3f47",
                                        "borderRadius": "8px",
                                        "overflow": "hidden",
                                    },
                                ),
                            ],
                        ),
                        dmc.Badge(
                            "All geos",
                            color="gray",
                            variant="light",
                            radius="sm",
                            leftSection=DashIconify(icon="tabler:map-pin", width=12),
                        ),
                        dmc.Badge(
                            "Demo",
                            color="teal",
                            variant="light",
                            radius="sm",
                            leftSection=DashIconify(icon="tabler:flask", width=12),
                        ),
                    ],
                ),
            ],
        ),
    )


def _navbar() -> dmc.AppShellNavbar:
    return dmc.AppShellNavbar(
        p="md",
        children=dmc.Stack(
            gap="xs",
            children=[
                dmc.Text(
                    "Navigate",
                    size="xs",
                    c="dimmed",
                    tt="uppercase",
                    fw=600,
                    mb=4,
                ),
                *[_nav_link(path, label, icon) for path, label, icon in ROUTES],
                dmc.Divider(my="md"),
                dmc.Text(
                    "About",
                    size="xs",
                    c="dimmed",
                    tt="uppercase",
                    fw=600,
                ),
                dmc.Text(
                    "Bayesian MMM (pymc-marketing) with geometric adstock, logistic "
                    "saturation, yearly Fourier seasonality, and a linear trend. "
                    "Fit with PyMC/NUTS; posterior cached to disk.",
                    size="xs",
                    c="dimmed",
                ),
            ],
        ),
    )


def create_app() -> Dash:
    app = Dash(
        __name__,
        title="MMM Demo",
        update_title=None,
        suppress_callback_exceptions=True,
    )

    results_by_geo = build_model_cache()

    base_result = results_by_geo["All"]
    app.layout = dmc.MantineProvider(
        theme=_theme(),
        forceColorScheme="dark",
        children=dmc.AppShell(
            header={"height": 64},
            navbar={"width": 260, "breakpoint": "sm"},
            padding="lg",
            children=[
                dcc.Location(id="url", refresh=False),
                _header(base_result),
                _navbar(),
                dmc.AppShellMain(
                    dmc.Container(
                        id="page-content",
                        fluid=True,
                        px=0,
                        style={"maxWidth": "1440px", "margin": "0 auto"},
                    )
                ),
            ],
        ),
    )

    _register_shell_callbacks(app, results_by_geo)
    register_overview_callbacks(app, results_by_geo)
    register_contributions_callbacks(app, results_by_geo)
    register_response_curve_callbacks(app, results_by_geo)
    register_optimiser_callbacks(app, results_by_geo)

    return app


# ---------- shell callbacks -----------------------------------------------


def _register_shell_callbacks(app: Dash, results_by_geo: dict[str, ModelResult]) -> None:
    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
    )
    def _render(pathname: str):
        result = results_by_geo["All"]
        if pathname == "/contributions":
            return build_contributions(result)
        if pathname == "/response-curves":
            return build_response_curves(result)
        if pathname == "/optimiser":
            return build_optimiser(result)
        return build_overview(result)

    @app.callback(
        *[Output({"type": "nav", "path": p}, "active") for p, _, _ in ROUTES],
        Input("url", "pathname"),
    )
    def _active(pathname: str):
        pathname = pathname or "/"
        return tuple(
            p == pathname or (p == "/" and pathname in ("", "/"))
            for p, _, _ in ROUTES
        )


if __name__ == "__main__":
    app = create_app()
    # use_reloader=False: debug mode otherwise spawns a second process; both can
    # bind :8050 after restarts, and the MMM would load twice on cold start.
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=8050)

"""MMM Demo Dashboard - Dash + DMC entrypoint.

Run with: `python app.py` (serves on http://127.0.0.1:8050).
"""

from __future__ import annotations

import threading

import dash_mantine_components as dmc
import pandas as pd
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from components.ids import (
    MODEL_REFRESH_STORE,
    OPT_DRAWS,
    OPT_REFIT_BTN,
    OPT_REFIT_STATUS,
    OPT_TARGET_ACCEPT,
    OPT_TUNE,
    OVERVIEW_DATE_STORE,
    REFIT_JOB_STORE,
    REFIT_OVERLAY_ROOT,
    REFIT_OVERLAY_STORE,
    REFIT_POLL_INTERVAL,
    REFIT_PROGRESS_CHAINS,
)
from data.loader import aggregate_geo, load_meridian, select_demo_geo
from model.mmm import ModelResult, fit_surrogate, save_sampler_config
from model.sampling_progress import SamplingProgressTracker
from pages import (
    build_contributions,
    build_optimiser,
    build_overview,
    build_overview_toolbar,
    build_response_curves,
    register_contributions_callbacks,
    register_optimiser_callbacks,
    register_overview_callbacks,
    register_response_curve_callbacks,
)

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
    """Fit (or load from disk cache) the Bayesian MMM on one demo geography.

    First launch samples with PyMC/NUTS (~60-90s, cached to `data/mmm_idata.nc`).
    Subsequent launches load the cached inference data and are near-instant.
    Sampler settings default from `model.mmm.load_sampler_config()` (optional JSON
    at `data/mmm_sampler_config.json`). Use the header **Options** panel to
    refit with new settings.
    """
    df = load_meridian()
    demo_geo = select_demo_geo(df)
    print(
        f"Loading MMM for {demo_geo} (first run ~60-90s while PyMC samples)...",
        flush=True,
    )
    result = fit_surrogate(aggregate_geo(df, demo_geo), demo_geo)
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
    sc = result.sampler_config or {}
    ta = float(sc.get("target_accept", 0.99))
    draws = int(sc.get("draws", 1500))
    tune = int(sc.get("tune", 3000))
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
                                    f"Bayesian MMM (pymc-marketing) on {result.geo}",
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
                                                description="0.75–0.999",
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
                                                color="teal",
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
                        dmc.Badge(
                            result.geo,
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


def _overview_date_store(result: ModelResult) -> dcc.Store:
    """Overview-only date range; other routes always use the full fitted period."""
    dmin = pd.to_datetime(result.dates).min()
    dmax = pd.to_datetime(result.dates).max()
    return dcc.Store(
        id=OVERVIEW_DATE_STORE,
        data={
            "start": dmin.date().isoformat(),
            "end": dmax.date().isoformat(),
        },
        storage_type="session",
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
                    "saturation, yearly Fourier seasonality, and controls. "
                    "Fit with PyMC/NUTS, posterior cached to disk.",
                    size="xs",
                    c="dimmed",
                ),
            ],
        ),
    )


def _refit_progress_placeholder() -> dmc.Stack:
    return dmc.Stack(
        gap=6,
        children=[
            dmc.Text(
                "Starting sampler…",
                size="xs",
                c="dimmed",
            ),
            dmc.Progress(
                value=None,
                striped=True,
                animated=True,
                color="teal",
                size="sm",
                radius="xl",
            ),
        ],
    )


def _refit_progress_from_snapshot(snapshot: dict | None) -> dmc.Stack:
    """Per-chain bars driven by `SamplingProgressTracker.snapshot()`."""
    if not snapshot or not snapshot.get("chains"):
        return _refit_progress_placeholder()
    phase = snapshot.get("phase") or "idle"
    if phase == "nuts" and snapshot.get("nuts_indeterminate"):
        return dmc.Stack(
            gap=6,
            children=[
                dmc.Text(
                    "NUTS sampling (nutpie / JAX)…",
                    size="xs",
                    c="teal",
                ),
                dmc.Progress(
                    value=None,
                    striped=True,
                    animated=True,
                    color="teal",
                    size="sm",
                    radius="xl",
                ),
                dmc.Text(
                    "Per-chain progress is only available with the PyMC sampler; "
                    "refits use nutpie so sampling matches the stable cold-fit path.",
                    size="xs",
                    c="dimmed",
                ),
            ],
        )
    parts: list = []
    if phase == "ppc":
        parts.append(
            dmc.Text(
                "Posterior predictive sampling (single progress in terminal)…",
                size="xs",
                c="teal",
            )
        )
        parts.append(
            dmc.Progress(
                value=None,
                striped=True,
                animated=True,
                color="teal",
                size="sm",
                radius="xl",
            )
        )
    rows: list = []
    for c in snapshot["chains"]:
        warm = "Warmup" if c.get("warmup") else "Sampling"
        pct = float(c.get("pct") or 0.0)
        rows.append(
            dmc.Stack(
                gap=4,
                children=[
                    dmc.Group(
                        justify="space-between",
                        wrap="nowrap",
                        children=[
                            dmc.Text(
                                f"Chain {int(c['chain']) + 1} · {warm}",
                                size="xs",
                                fw=500,
                            ),
                            dmc.Text(f"{pct:.0f}%", size="xs", c="dimmed"),
                        ],
                    ),
                    dmc.Progress(
                        value=pct,
                        color="teal",
                        size="sm",
                        radius="xl",
                    ),
                ],
            )
        )
    parts.extend(rows)
    if not snapshot.get("nuts_indeterminate"):
        parts.append(
            dmc.Text(
                f"Mean across chains ≈ {float(snapshot.get('overall_pct') or 0):.0f}% "
                "(matches PyMC split progress in the terminal).",
                size="xs",
                c="dimmed",
            )
        )
    return dmc.Stack(gap="sm", children=parts)


def _refit_overlay_root() -> dmc.Box:
    """Full-viewport blocking layer while NUTS refits (paired with REFIT_OVERLAY_STORE)."""
    return dmc.Box(
        id=REFIT_OVERLAY_ROOT,
        style={
            "position": "fixed",
            "inset": 0,
            "zIndex": 10000,
            "display": "none",
            "alignItems": "center",
            "justifyContent": "center",
            "flexDirection": "column",
            "backgroundColor": "rgba(10, 12, 16, 0.72)",
            "backdropFilter": "blur(8px)",
        },
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
                                    dmc.Text("Refitting model", fw=600, size="md"),
                                    dmc.Text(
                                        "PyMC/NUTS is sampling. This usually takes several minutes.",
                                        size="sm",
                                        c="dimmed",
                                    ),
                                ],
                            ),
                            dmc.Loader(color="teal", size="md", type="oval"),
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
                                children=_refit_progress_placeholder(),
                            ),
                            dmc.Text(
                                "Native PyMC NUTS reports each chain here; the same split "
                                "bars appear in the terminal where you ran `python app.py`.",
                                size="xs",
                                c="dimmed",
                            ),
                        ],
                    ),
                ],
            ),
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
        children=dmc.Box(
            pos="relative",
            style={"minHeight": "100vh"},
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
                        _overview_date_store(base_result),
                        _header(base_result),
                        _navbar(),
                        dmc.AppShellMain(
                            dmc.Stack(
                                gap=0,
                                children=[
                                    build_overview_toolbar(base_result),
                                    dmc.Container(
                                        id="page-content",
                                        fluid=True,
                                        px=0,
                                        style={
                                            "maxWidth": "1440px",
                                            "margin": "0 auto",
                                        },
                                        children=build_overview(base_result),
                                    ),
                                ],
                            )
                        ),
                    ],
                ),
                _refit_overlay_root(),
            ],
        ),
    )

    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) {
                return window.dash_clientside.no_update;
            }
            return [{ open: true }, true];
        }
        """,
        Output(REFIT_OVERLAY_STORE, "data"),
        Output(OPT_REFIT_BTN, "loading"),
        Input(OPT_REFIT_BTN, "n_clicks"),
        prevent_initial_call=True,
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
        Output(REFIT_OVERLAY_ROOT, "style"),
        Input(REFIT_OVERLAY_STORE, "data"),
    )
    def _sync_refit_overlay_display(data: dict | None):
        base = {
            "position": "fixed",
            "inset": 0,
            "zIndex": 10000,
            "alignItems": "center",
            "justifyContent": "center",
            "flexDirection": "column",
            "backgroundColor": "rgba(10, 12, 16, 0.72)",
            "backdropFilter": "blur(8px)",
        }
        if isinstance(data, dict) and data.get("open"):
            return {**base, "display": "flex"}
        return {**base, "display": "none"}

    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
        Input(MODEL_REFRESH_STORE, "data"),
    )
    def _render(pathname: str, _refresh: int | None):
        result = results_by_geo["All"]
        if pathname == "/contributions":
            return build_contributions(result)
        if pathname == "/response-curves":
            return build_response_curves(result)
        if pathname == "/optimiser":
            return build_optimiser(result)
        return build_overview(result)

    _refit = {
        "thread": None,
        "tracker": None,
        "err": None,
        "cfg": None,
        "lock": threading.Lock(),
    }

    @app.callback(
        Output(REFIT_POLL_INTERVAL, "disabled"),
        Output(REFIT_PROGRESS_CHAINS, "children"),
        Output(REFIT_JOB_STORE, "data"),
        Input(OPT_REFIT_BTN, "n_clicks"),
        State(OPT_TARGET_ACCEPT, "value"),
        State(OPT_DRAWS, "value"),
        State(OPT_TUNE, "value"),
        prevent_initial_call=True,
    )
    def _refit_start(
        n_clicks: int | None,
        ta: float | None,
        draws: int | None,
        tune: int | None,
    ):
        if not n_clicks:
            raise PreventUpdate
        cfg = {
            "target_accept": float(ta if ta is not None else 0.95),
            "draws": int(draws if draws is not None else 1500),
            "tune": int(tune if tune is not None else 3000),
        }
        cfg["draws"] = max(100, cfg["draws"])
        cfg["tune"] = max(200, cfg["tune"])
        cfg["target_accept"] = min(0.9999, max(0.75, cfg["target_accept"]))
        with _refit["lock"]:
            th = _refit["thread"]
            if th is not None and th.is_alive():
                raise PreventUpdate
        save_sampler_config(cfg)
        print("Refitting MMM with sampler config:", cfg, flush=True)
        raw_df = load_meridian()
        demo_geo = select_demo_geo(raw_df)
        df = aggregate_geo(raw_df, demo_geo)
        tracker = SamplingProgressTracker()
        err_box: list[BaseException | None] = [None]

        def work() -> None:
            try:
                results_by_geo["All"] = fit_surrogate(
                    df, demo_geo, cfg, progress=tracker
                )
            except Exception as exc:
                err_box[0] = exc

        thread = threading.Thread(target=work, daemon=True)
        with _refit["lock"]:
            _refit["cfg"] = cfg
            _refit["tracker"] = tracker
            _refit["err"] = err_box
            _refit["thread"] = thread
        thread.start()
        return (
            False,
            _refit_progress_from_snapshot(tracker.snapshot()),
            {"running": True},
        )

    @app.callback(
        Output(REFIT_PROGRESS_CHAINS, "children", allow_duplicate=True),
        Output(MODEL_REFRESH_STORE, "data", allow_duplicate=True),
        Output(OPT_REFIT_STATUS, "children"),
        Output(OPT_TARGET_ACCEPT, "value"),
        Output(OPT_DRAWS, "value"),
        Output(OPT_TUNE, "value"),
        Output(REFIT_OVERLAY_STORE, "data", allow_duplicate=True),
        Output(OPT_REFIT_BTN, "loading", allow_duplicate=True),
        Output(REFIT_POLL_INTERVAL, "disabled", allow_duplicate=True),
        Output(REFIT_JOB_STORE, "data", allow_duplicate=True),
        Input(REFIT_POLL_INTERVAL, "n_intervals"),
        State(MODEL_REFRESH_STORE, "data"),
        prevent_initial_call=True,
    )
    def _poll_refit(n_intervals: int | None, gen: int | None):
        if not n_intervals:
            raise PreventUpdate
        with _refit["lock"]:
            thread = _refit["thread"]
            tracker = _refit["tracker"]
            err_box = _refit["err"]
            cfg = _refit["cfg"]
        if thread is None:
            raise PreventUpdate
        snap = tracker.snapshot() if tracker is not None else None
        children = _refit_progress_from_snapshot(snap)
        if thread.is_alive():
            return (
                children,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )
        err = err_box[0] if err_box else None
        with _refit["lock"]:
            _refit["thread"] = None
            _refit["tracker"] = None
            _refit["err"] = None
            _refit["cfg"] = None
        if err is not None:
            status = dmc.Stack(
                gap=4,
                children=[
                    dmc.Text(f"Refit failed: {err}", size="xs", c="red"),
                ],
            )
            return (
                children,
                no_update,
                status,
                no_update,
                no_update,
                no_update,
                {"open": False},
                False,
                True,
                {"running": False},
            )
        done_cfg = cfg if cfg is not None else {}
        status = dmc.Stack(
            gap=4,
            children=[
                dmc.Text("Refit complete.", size="xs", c="teal"),
                dmc.Text(
                    "Charts reloaded. Sampling settings saved to data/mmm_sampler_config.json.",
                    size="xs",
                    c="dimmed",
                ),
            ],
        )
        return (
            children,
            (gen or 0) + 1,
            status,
            done_cfg.get("target_accept", 0.95),
            done_cfg.get("draws", 1500),
            done_cfg.get("tune", 3000),
            {"open": False},
            False,
            True,
            {"running": False},
        )

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

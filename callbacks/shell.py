"""Callbacks for navigation state and model refits."""

from __future__ import annotations

import threading

import dash_mantine_components as dmc
from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

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
from dashboard.theme import POSITIVE_COLOR, REFIT_OVERLAY_STYLE
from data.loader import aggregate_geo, load_meridian, select_demo_geo
from layouts.shell import nav_pages, refit_progress_from_snapshot
from model.mmm import ModelResult, fit_surrogate, save_sampler_config
from model.sampling_progress import SamplingProgressTracker


def register_shell_callbacks(app, results_by_geo: dict[str, ModelResult]) -> None:
    @app.callback(
        Output(REFIT_OVERLAY_ROOT, "style"),
        Input(REFIT_OVERLAY_STORE, "data"),
    )
    def _sync_refit_overlay_display(data: dict | None):
        if isinstance(data, dict) and data.get("open"):
            return {**REFIT_OVERLAY_STYLE, "display": "flex"}
        return {**REFIT_OVERLAY_STYLE, "display": "none"}

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
            refit_progress_from_snapshot(tracker.snapshot()),
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
        children = refit_progress_from_snapshot(snap)
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
                children=[dmc.Text(f"Refit failed: {err}", size="xs", c="red")],
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
                dmc.Text("Refit complete.", size="xs", c=POSITIVE_COLOR),
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

    pages = nav_pages()

    @app.callback(
        *[Output({"type": "nav", "path": page["path"]}, "active") for page in pages],
        Input("url", "pathname"),
    )
    def _active(pathname: str | None):
        pathname = pathname or "/"
        return tuple(
            page["path"] == pathname or (page["path"] == "/" and pathname in ("", "/"))
            for page in pages
        )

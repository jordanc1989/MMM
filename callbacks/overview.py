"""Callbacks for the overview page."""

from __future__ import annotations

import pandas as pd
from dash import Input, Output
from dash.exceptions import PreventUpdate

from components.ids import (
    ACTUAL_PREDICTED_GRAPH_ID,
    MODEL_REFRESH_STORE,
    OVERVIEW_DATE_STORE,
    OVERVIEW_KPI_GRID_ID,
    OVERVIEW_RANGE_PRESET,
    OVERVIEW_RESIDUALS_ID,
    OVERVIEW_TOOLBAR,
    OVERVIEW_WATERFALL_ID,
    OVERVIEW_YEAR_SELECT,
)
from dashboard.theme import OVERVIEW_TOOLBAR_STYLE
from figures.overview import (
    actual_vs_predicted_chart,
    residuals_diagnostic_figure,
    revenue_waterfall,
)
from layouts.overview import _bounds, _build_kpi_grid, _range_from_preset
from model.mmm import ModelResult, slice_model_result

def register_overview_callbacks(app, results_by_geo: dict[str, ModelResult]) -> None:
    @app.callback(
        Output(OVERVIEW_TOOLBAR, "style"),
        Input("url", "pathname"),
    )
    def _overview_toolbar_visibility(pathname: str | None):
        pathname = pathname or "/"
        if pathname in ("/", ""):
            return {**OVERVIEW_TOOLBAR_STYLE, "display": "block"}
        return {**OVERVIEW_TOOLBAR_STYLE, "display": "none"}

    @app.callback(
        Output(OVERVIEW_DATE_STORE, "data"),
        Output(OVERVIEW_YEAR_SELECT, "style"),
        Input("url", "pathname"),
        Input(OVERVIEW_RANGE_PRESET, "value"),
        Input(OVERVIEW_YEAR_SELECT, "value"),
    )
    def _sync_range_from_presets(
        pathname: str | None, preset: str | None, year_sel: str | None
    ):
        pathname = pathname or "/"
        if pathname not in ("/", ""):
            raise PreventUpdate
        dmin, dmax = _bounds(results_by_geo["All"])
        p = preset or "full"
        s, e = _range_from_preset(p, year_sel, dmin, dmax)
        style = (
            {"display": "block", "minWidth": 140}
            if p == "year"
            else {"display": "none"}
        )
        return (
            {"start": s.date().isoformat(), "end": e.date().isoformat()},
            style,
        )

    @app.callback(
        Output(ACTUAL_PREDICTED_GRAPH_ID, "figure"),
        Output(OVERVIEW_KPI_GRID_ID, "children"),
        Output(OVERVIEW_WATERFALL_ID, "figure"),
        Output(OVERVIEW_RESIDUALS_ID, "figure"),
        Input("url", "pathname"),
        Input(OVERVIEW_DATE_STORE, "data"),
        Input(MODEL_REFRESH_STORE, "data"),
    )
    def _update_overview(pathname, data, _refresh):
        pathname = pathname or "/"
        if pathname not in ("/", ""):
            raise PreventUpdate
        base = results_by_geo["All"]
        dmin = pd.to_datetime(base.dates).min()
        dmax = pd.to_datetime(base.dates).max()
        if not data or not isinstance(data, dict):
            start, end = dmin, dmax
        else:
            raw_s, raw_e = data.get("start"), data.get("end")
            if raw_s is None or raw_e is None:
                start, end = dmin, dmax
            else:
                start = pd.Timestamp(raw_s)
                end = pd.Timestamp(raw_e)
        start = max(start, dmin)
        end = min(end, dmax)
        if start > end:
            start, end = end, start

        sliced = slice_model_result(base, start, end)
        fig_pred = actual_vs_predicted_chart(sliced)
        kpis = _build_kpi_grid(sliced, base, start, end)
        fig_wf = revenue_waterfall(sliced)
        fig_res = residuals_diagnostic_figure(sliced)
        return fig_pred, kpis, fig_wf, fig_res

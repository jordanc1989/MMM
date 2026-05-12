"""Callbacks for response curves."""

from __future__ import annotations

from dash import Input, Output

from components.ids import MODEL_REFRESH_STORE
from figures.response_curves import response_curve_figure
from layouts.response_curves import CHANNEL_GRAPH_ID, CHANNEL_SELECT_ID, CHANNEL_STATS_ID, response_stats
from model.mmm import ModelResult

def register_response_curve_callbacks(app, results_by_geo: dict[str, ModelResult]) -> None:
    @app.callback(
        Output(CHANNEL_GRAPH_ID, "figure"),
        Output(CHANNEL_STATS_ID, "children"),
        Input(CHANNEL_SELECT_ID, "value"),
        Input(MODEL_REFRESH_STORE, "data"),
    )
    def _update(channel: str, _refresh: int | None):
        result = results_by_geo["All"]
        if not channel:
            channel = result.channels[0]
        return response_curve_figure(result, channel), response_stats(result, channel)

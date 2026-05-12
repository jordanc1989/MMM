"""Bayesian MMM Dashboard entrypoint.

Run with: `python app.py` (serves on http://127.0.0.1:8050).
"""

from __future__ import annotations

from dash import Dash, Input, Output

from callbacks.optimiser import register_optimiser_callbacks
from callbacks.overview import register_overview_callbacks
from callbacks.response_curves import register_response_curve_callbacks
from callbacks.shell import register_shell_callbacks
from components.ids import OPT_REFIT_BTN, REFIT_OVERLAY_STORE
from dashboard.data import build_model_cache
from dashboard.theme import APP_TITLE
from layouts.shell import build_shell
from pages.contributions import (
    register_contributions_callbacks,
)


def create_app() -> Dash:
    app = Dash(
        __name__,
        title=APP_TITLE,
        update_title=None,
        suppress_callback_exceptions=True,
    )

    results_by_geo = build_model_cache()
    app.layout = build_shell(results_by_geo["All"])

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

    register_shell_callbacks(app, results_by_geo)
    register_overview_callbacks(app, results_by_geo)
    register_contributions_callbacks(app, results_by_geo)
    register_response_curve_callbacks(app, results_by_geo)
    register_optimiser_callbacks(app, results_by_geo)

    return app


if __name__ == "__main__":
    app = create_app()
    # use_reloader=False: debug mode otherwise spawns a second process; both can
    # bind :8050 after restarts, and the MMM would load twice on cold start.
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=8050)

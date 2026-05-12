"""Compatibility exports for the response curves page."""

import dash

from callbacks.response_curves import register_response_curve_callbacks
from dashboard.state import current_result
from figures.response_curves import response_curve_figure
from layouts.response_curves import build_response_curves, response_stats

dash.register_page(
    __name__,
    path="/response-curves",
    name="Response Curves",
    title="Response Curves | Bayesian MMM Dashboard",
    order=2,
    icon="tabler:chart-ppf",
)


def layout(**_kwargs):
    return build_response_curves(current_result())


__all__ = [
    "build_response_curves",
    "register_response_curve_callbacks",
    "response_curve_figure",
    "response_stats",
]

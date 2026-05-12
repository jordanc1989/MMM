"""Compatibility exports for the overview page."""

import dash

from callbacks.overview import register_overview_callbacks
from dashboard.state import current_result
from figures.overview import actual_vs_predicted_chart, residuals_diagnostic_figure, revenue_waterfall
from layouts.overview import build_overview, build_overview_toolbar

dash.register_page(
    __name__,
    path="/",
    name="Overview",
    title="Overview | Bayesian MMM Dashboard",
    order=0,
    icon="tabler:layout-dashboard",
)


def layout(**_kwargs):
    return build_overview(current_result())


__all__ = [
    "actual_vs_predicted_chart",
    "build_overview",
    "build_overview_toolbar",
    "register_overview_callbacks",
    "residuals_diagnostic_figure",
    "revenue_waterfall",
]

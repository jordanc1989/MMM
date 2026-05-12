"""Compatibility exports for the optimiser page."""

import dash

from callbacks.optimiser import register_optimiser_callbacks
from dashboard.state import current_result
from layouts.optimiser import build_optimiser

dash.register_page(
    __name__,
    path="/optimiser",
    name="Optimiser",
    title="Optimiser | Bayesian MMM Dashboard",
    order=3,
    icon="tabler:adjustments-horizontal",
)


def layout(**_kwargs):
    return build_optimiser(current_result())


__all__ = ["build_optimiser", "register_optimiser_callbacks"]

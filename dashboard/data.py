"""Dashboard data loading and cache wiring."""

from __future__ import annotations

import pandas as pd
from dash import dcc

from components.ids import OVERVIEW_DATE_STORE
from data.loader import aggregate_geo, load_meridian, select_demo_geo
from model.mmm import ModelResult, fit_surrogate


def build_model_cache() -> dict[str, ModelResult]:
    """Fit or load the cached Bayesian MMM for the demo geography."""
    df = load_meridian()
    demo_geo = select_demo_geo(df)
    print(
        f"Loading MMM for {demo_geo} (first run ~60-90s while nutpie/JAX samples)...",
        flush=True,
    )
    result = fit_surrogate(aggregate_geo(df, demo_geo), demo_geo)
    print("MMM ready.", flush=True)
    return {"All": result}


def overview_date_store(result: ModelResult) -> dcc.Store:
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

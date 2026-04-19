"""Load Google Meridian's simulated sample dataset.

Downloads `geo_all_channels.csv` on first run, caches locally, and returns a
tidy long-format DataFrame with engineered columns (`revenue`, renamed
channels).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

MERIDIAN_CSV_URL = (
    "https://raw.githubusercontent.com/google/meridian/main/"
    "meridian/data/simulated_data/csv/geo_all_channels.csv"
)
DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "geo_all_channels.csv"

CHANNELS: list[str] = [
    "Search",
    "Social",
    "Video",
    "Display",
    "Audio",
]
"""Human-friendly channel names mapped 1:1 onto Meridian's Channel0..Channel4."""

CHANNEL_RAW_TO_LABEL = {f"Channel{i}": CHANNELS[i] for i in range(5)}

CONTROLS: list[str] = [
    "competitor_sales_control",
    "sentiment_score_control",
    "Promo",
    "Organic_channel0_impression",
]

TREND_COLUMNS: list[str] = ["t"]
"""Column names injected by `aggregate_geo` to act as a linear-trend regressor."""


def _download_if_missing() -> None:
    if CSV_PATH.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MERIDIAN_CSV_URL, CSV_PATH)


def load_meridian() -> pd.DataFrame:
    """Return a tidy DataFrame indexed by (geo, time) with renamed channels.

    Columns include:
      - `geo`, `time` (datetime)
      - `<Channel>_spend`, `<Channel>_impression` for each channel in CHANNELS
      - `revenue` (conversions * revenue_per_conversion)
      - the control columns in CONTROLS
      - `population`
    """
    _download_if_missing()
    df = pd.read_csv(CSV_PATH, index_col=0)
    df["time"] = pd.to_datetime(df["time"])

    for raw, label in CHANNEL_RAW_TO_LABEL.items():
        df.rename(
            columns={
                f"{raw}_spend": f"{label}_spend",
                f"{raw}_impression": f"{label}_impression",
            },
            inplace=True,
        )

    df["revenue"] = df["conversions"] * df["revenue_per_conversion"]

    return df.sort_values(["geo", "time"]).reset_index(drop=True)


def aggregate_geo(df: pd.DataFrame, geo: str | None) -> pd.DataFrame:
    """Slice the dataframe to a geo, or aggregate across all geos.

    Aggregation sums spend, impressions, conversions, and revenue across geos
    while taking spend-weighted means of control variables.
    """
    if geo and geo != "All":
        out = df[df["geo"] == geo].sort_values("time").reset_index(drop=True)
    else:
        sum_cols = [c for c in df.columns if c.endswith(("_spend", "_impression"))]
        sum_cols += ["conversions", "revenue"]

        out = df.groupby("time", as_index=False).agg(
            {
                **{c: "sum" for c in sum_cols},
                **{c: "mean" for c in CONTROLS},
                "revenue_per_conversion": "mean",
                "population": "sum",
            }
        )
        out["geo"] = "All"
        out = out.sort_values("time").reset_index(drop=True)

    n = len(out)
    out["t"] = (np.arange(n) - (n - 1) / 2.0) / max(n, 1)
    return out

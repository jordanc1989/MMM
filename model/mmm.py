"""Bayesian MMM backed by `pymc-marketing`.

Architecture:
    Revenue_t = intercept
              + linear_trend_t              (via the `t` control column)
              + yearly_seasonality_t        (Fourier, n_order=3)
              + sum_k gamma_k * control_k_t (economic controls)
              + sum_c beta_c * LogisticSaturation(
                        GeometricAdstock(spend_{c,t}, alpha_c, l_max=8),
                        lam_c,
                    )

The heavy lifting is done by `pymc_marketing.mmm.MMM` which samples with NUTS.
The fitted `InferenceData` is cached to `data/mmm_idata.nc` so only the first
launch pays the sampling cost.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import CHANNELS, CONTROLS, TREND_COLUMNS

# --- pytensor / macOS SDK preflight --------------------------------------
# pymc-marketing -> pymc -> pytensor JIT-compiles C++; clang needs SDKROOT on
# recent macOS to find <vector> et al.  Setting it here (before pytensor is
# imported) keeps the rest of the app ignorant of this detail.
if sys.platform == "darwin" and "SDKROOT" not in os.environ:
    try:
        sdk = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True, timeout=5
        ).strip()
        if sdk:
            os.environ["SDKROOT"] = sdk
    except (OSError, subprocess.SubprocessError):
        pass

_COMPILEDIR = Path(__file__).resolve().parent.parent / ".pytensor_cache"
_COMPILEDIR.mkdir(exist_ok=True)
_existing = os.environ.get("PYTENSOR_FLAGS", "")
if "base_compiledir" not in _existing:
    os.environ["PYTENSOR_FLAGS"] = (
        f"{_existing},base_compiledir={_COMPILEDIR}".lstrip(",")
    )

from pymc_marketing.mmm import (  # noqa: E402
    GeometricAdstock,
    LogisticSaturation,
)
from pymc_marketing.mmm.multidimensional import MMM  # noqa: E402

# ---- constants ----------------------------------------------------------

CHANNEL_COLUMNS: list[str] = [f"{c}_spend" for c in CHANNELS]
CONTROL_COLUMNS: list[str] = CONTROLS + TREND_COLUMNS
ADSTOCK_L_MAX = 8
FOURIER_N_ORDER = 3

IDATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mmm_idata.nc"


# ---- result container ---------------------------------------------------


@dataclass
class ModelResult:
    geo: str
    channels: list[str]
    dates: np.ndarray
    spend: dict[str, np.ndarray]
    revenue: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray

    # Posterior-mean time series, original scale
    contributions: dict[str, np.ndarray]
    control_contributions: dict[str, np.ndarray]
    intercept_contribution: np.ndarray
    trend_contribution: np.ndarray
    seasonality_contribution: np.ndarray
    baseline: np.ndarray  # intercept + trend + seasonality + controls

    # Posterior-mean parameters (used for response curves / optimiser)
    betas: dict[str, float]
    decays: dict[str, float]
    saturation_lam: dict[str, float]
    channel_scale: dict[str, float]
    target_scale: float

    intercept: float
    r2: float
    mape: float
    r2_hdi: tuple[float, float]

    # 94% HDI (low, high) bands per channel contribution time series
    contribution_hdi: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )

    # Backwards-compat shim; older code referenced `saturation_k`.
    @property
    def saturation_k(self) -> dict[str, float]:
        return self.saturation_lam

    # Backwards-compat shim; older code referenced `control_coef`.
    @property
    def control_coef(self) -> dict[str, float]:
        # Return effective average-per-unit effect for each control.
        out: dict[str, float] = {}
        for k, series in self.control_contributions.items():
            out[k] = float(series.mean())
        return out

    @property
    def total_spend(self) -> dict[str, float]:
        return {c: float(self.spend[c].sum()) for c in self.channels}

    @property
    def total_contribution(self) -> dict[str, float]:
        return {c: float(self.contributions[c].sum()) for c in self.channels}

    @property
    def total_revenue(self) -> float:
        return float(self.revenue.sum())

    @property
    def roi(self) -> dict[str, float]:
        roi = {}
        for c in self.channels:
            s = self.total_spend[c]
            roi[c] = float(self.total_contribution[c] / s) if s > 0 else 0.0
        return roi

    @property
    def n_weeks(self) -> int:
        return int(len(self.dates))


# ---- helpers ------------------------------------------------------------


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _mape(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = np.abs(y) > 1e-9
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])))


def _hdi(samples: np.ndarray, prob: float = 0.94) -> tuple[float, float]:
    """Simple empirical highest-density-interval."""
    s = np.sort(np.asarray(samples).ravel())
    n = len(s)
    if n == 0:
        return 0.0, 0.0
    k = max(1, int(np.floor(prob * n)))
    width = s[k - 1 :] - s[: n - k + 1]
    if len(width) == 0:
        return float(s[0]), float(s[-1])
    i = int(np.argmin(width))
    return float(s[i]), float(s[i + k - 1])


def _channel_hdi_bands(
    posterior, channel: str, target_scale: float, prob: float = 0.94
) -> tuple[np.ndarray, np.ndarray]:
    """Return (low, high) arrays along `date` for a given channel's contribution.

    In the multidimensional MMM the posterior stores `channel_contribution` in
    **scaled** space; we multiply by `target_scale` to return currency units.
    """
    arr = (
        posterior["channel_contribution"]
        .sel(channel=channel)
        .stack(sample=("chain", "draw"))
        .transpose("date", "sample")
        .to_numpy()
        * target_scale
    )  # (n_dates, n_samples)
    alpha = (1 - prob) / 2
    low = np.quantile(arr, alpha, axis=1)
    high = np.quantile(arr, 1 - alpha, axis=1)
    return low, high


# ---- fit / load ---------------------------------------------------------


def _build_mmm() -> MMM:
    return MMM(
        date_column="time",
        channel_columns=CHANNEL_COLUMNS,
        target_column="revenue",
        control_columns=CONTROL_COLUMNS,
        adstock=GeometricAdstock(l_max=ADSTOCK_L_MAX),
        saturation=LogisticSaturation(),
        yearly_seasonality=FOURIER_N_ORDER,
    )


def _fit_mmm(df: pd.DataFrame) -> MMM:
    mmm = _build_mmm()
    X = df[["time", *CHANNEL_COLUMNS, *CONTROL_COLUMNS]]
    y = df["revenue"]
    mmm.fit(
        X,
        y,
        draws=500,
        tune=500,
        chains=2,
        target_accept=0.9,
        random_seed=42,
        progressbar=False,
    )
    mmm.sample_posterior_predictive(X, extend_idata=True, progressbar=False)
    return mmm


def _load_or_fit(df: pd.DataFrame) -> MMM:
    if IDATA_PATH.exists():
        try:
            return MMM.load(str(IDATA_PATH))
        except Exception:  # noqa: BLE001
            # Corrupt / incompatible cache — refit.
            try:
                IDATA_PATH.unlink()
            except OSError:
                pass

    print(
        "Fitting pymc-marketing MMM (first run ~60s, cached after)...",
        flush=True,
    )
    mmm = _fit_mmm(df)
    try:
        IDATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        mmm.save(str(IDATA_PATH))
    except Exception as exc:  # noqa: BLE001
        print(f"(warning) failed to cache InferenceData: {exc}", flush=True)
    return mmm


# ---- result extraction --------------------------------------------------


def _extract_scalars(mmm: MMM) -> tuple[dict[str, float], float]:
    """Pull per-channel and target max-abs scale factors from the fitted MMM.

    The multidimensional MMM exposes them as an xarray-friendly dict via
    `get_scales_as_xarray`, so we no longer rely on the (removed) sklearn
    `channel_transformer` / `target_transformer` pipelines.
    """
    scales = mmm.get_scales_as_xarray()
    channel_da = scales["channel_scale"]
    channel_scale = {
        str(ch): float(channel_da.sel(channel=ch).item())
        for ch in channel_da.coords["channel"].values
    }
    target_scale = float(scales["target_scale"].item())
    return channel_scale, target_scale


def _posterior_mean_params(mmm: MMM) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    post = mmm.idata.posterior
    alphas = post["adstock_alpha"].mean(("chain", "draw"))
    lams = post["saturation_lam"].mean(("chain", "draw"))
    betas = post["saturation_beta"].mean(("chain", "draw"))

    decays: dict[str, float] = {}
    lam_map: dict[str, float] = {}
    beta_map: dict[str, float] = {}
    for ch in mmm.channel_columns:
        # coords on "channel" use the original channel-column names
        decays[ch] = float(alphas.sel(channel=ch).item())
        lam_map[ch] = float(lams.sel(channel=ch).item())
        beta_map[ch] = float(betas.sel(channel=ch).item())
    return decays, lam_map, beta_map


def _r2_hdi(
    y_true: np.ndarray, y_draws: np.ndarray, prob: float = 0.94
) -> tuple[float, float]:
    """HDI of R^2 computed per posterior sample.

    `y_draws` is (n_samples, n_dates)."""
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0, 0.0
    ss_res = np.sum((y_true[None, :] - y_draws) ** 2, axis=1)
    r2_samples = 1.0 - ss_res / ss_tot
    return _hdi(r2_samples, prob=prob)


def _r2_hdi_from_components(
    post, y_true: np.ndarray, target_scale: float, prob: float = 0.94
) -> tuple[float, float]:
    """Reconstruct the model-implied mean per draw and return its R^2 HDI.

    We sum contribution variables (all in scaled space) per draw, scale back
    to currency units, then score against observed revenue. Keeps the HDI
    focused on parameter uncertainty rather than observation noise.
    """
    channel = post["channel_contribution"].sum("channel")  # (chain, draw, date)
    control = post["control_contribution"].sum("control")
    season = post["yearly_seasonality_contribution"]
    intercept = post["intercept_contribution"]  # (chain, draw)
    total_scaled = (channel + control + season) + intercept  # broadcasts scalar
    y_draws = (
        total_scaled.stack(sample=("chain", "draw"))
        .transpose("sample", "date")
        .to_numpy()
        * target_scale
    )
    return _r2_hdi(y_true, y_draws, prob=prob)


def _r2_hdi_from_components_masked(
    post,
    y_true: np.ndarray,
    target_scale: float,
    mask: np.ndarray,
    prob: float = 0.94,
) -> tuple[float, float]:
    """Same as `_r2_hdi_from_components` but only dates where `mask` is True."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0.0, 0.0
    channel = post["channel_contribution"].sum("channel").isel(date=idx)
    control = post["control_contribution"].sum("control").isel(date=idx)
    season = post["yearly_seasonality_contribution"].isel(date=idx)
    intercept = post["intercept_contribution"]
    total_scaled = (channel + control + season) + intercept
    y_draws = (
        total_scaled.stack(sample=("chain", "draw"))
        .transpose("sample", "date")
        .to_numpy()
        * target_scale
    )
    y_sel = y_true[mask]
    return _r2_hdi(y_sel, y_draws, prob=prob)


def _build_result(mmm: MMM, df: pd.DataFrame, geo_label: str) -> ModelResult:
    df = df.sort_values("time").reset_index(drop=True)
    dates = df["time"].values
    revenue = df["revenue"].to_numpy(dtype=float)

    spend = {
        c: df[f"{c}_spend"].to_numpy(dtype=float) for c in CHANNELS
    }

    post = mmm.idata.posterior
    channel_scale, target_scale = _extract_scalars(mmm)

    # Contribution variables in the multidimensional MMM live in **scaled**
    # space; multiply by target_scale to return to currency units.
    channel_contribution_scaled = post["channel_contribution"].mean(
        ("chain", "draw")
    )  # (date, channel)
    control_contribution_scaled = post["control_contribution"].mean(
        ("chain", "draw")
    )  # (date, control)
    seasonality_contribution = (
        post["yearly_seasonality_contribution"]
        .mean(("chain", "draw"))
        .to_numpy()
        * target_scale
    )  # (date,)
    intercept_scaled_mean = float(
        post["intercept_contribution"].mean(("chain", "draw")).item()
    )

    intercept_series = np.full(len(df), intercept_scaled_mean * target_scale)

    contributions: dict[str, np.ndarray] = {}
    for i, c in enumerate(CHANNELS):
        col = CHANNEL_COLUMNS[i]
        contributions[c] = (
            channel_contribution_scaled.sel(channel=col).to_numpy() * target_scale
        )

    contribution_hdi: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, c in enumerate(CHANNELS):
        col = CHANNEL_COLUMNS[i]
        contribution_hdi[c] = _channel_hdi_bands(post, col, target_scale)

    trend_col = TREND_COLUMNS[0]
    trend_contribution = (
        control_contribution_scaled.sel(control=trend_col).to_numpy() * target_scale
    )

    control_contributions: dict[str, np.ndarray] = {}
    for k in CONTROLS:
        control_contributions[k] = (
            control_contribution_scaled.sel(control=k).to_numpy() * target_scale
        )

    control_total = np.sum(
        [control_contributions[k] for k in CONTROLS], axis=0
    )
    baseline = (
        intercept_series
        + trend_contribution
        + seasonality_contribution
        + control_total
    )
    channel_total = np.sum([contributions[c] for c in CHANNELS], axis=0)
    fitted = baseline + channel_total
    residuals = revenue - fitted

    decays, lam_map_by_col, beta_map_by_col = _posterior_mean_params(mmm)
    decays_out = {c: decays[CHANNEL_COLUMNS[i]] for i, c in enumerate(CHANNELS)}
    lam_out = {c: lam_map_by_col[CHANNEL_COLUMNS[i]] for i, c in enumerate(CHANNELS)}
    betas_out = {c: beta_map_by_col[CHANNEL_COLUMNS[i]] for i, c in enumerate(CHANNELS)}
    scale_out = {c: channel_scale[CHANNEL_COLUMNS[i]] for i, c in enumerate(CHANNELS)}

    # R^2 HDI from the model-implied mean: sum contribution components per
    # draw, back-scale to currency, and compute R^2 against observed revenue.
    r2_hdi = _r2_hdi_from_components(post, revenue, target_scale)

    return ModelResult(
        geo=geo_label,
        channels=list(CHANNELS),
        dates=dates,
        spend=spend,
        revenue=revenue,
        fitted=fitted,
        residuals=residuals,
        contributions=contributions,
        control_contributions=control_contributions,
        intercept_contribution=intercept_series,
        trend_contribution=trend_contribution,
        seasonality_contribution=seasonality_contribution,
        baseline=baseline,
        betas=betas_out,
        decays=decays_out,
        saturation_lam=lam_out,
        channel_scale=scale_out,
        target_scale=target_scale,
        intercept=float(intercept_series[0]),
        r2=_r2(revenue, fitted),
        mape=_mape(revenue, fitted),
        r2_hdi=r2_hdi,
        contribution_hdi=contribution_hdi,
    )


def fit_surrogate(df: pd.DataFrame, geo_label: str = "All") -> ModelResult:
    """Fit (or load from disk cache) the Bayesian MMM and assemble ModelResult."""
    mmm = _load_or_fit(df)
    return _build_result(mmm, df, geo_label)


# ---- deterministic forward pass (for curves / optimiser) -----------------


def _logistic_saturation(x: np.ndarray | float, lam: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (1.0 - np.exp(-lam * x)) / (1.0 + np.exp(-lam * x))


def _steady_state_contribution(
    weekly_spend: np.ndarray | float,
    *,
    lam: float,
    beta: float,
    channel_scale: float,
    target_scale: float,
) -> np.ndarray | float:
    """Predicted weekly contribution in **original units** for a constant spend.

    Geometric adstock with `normalize=True` converges to the input itself at
    steady state (the normalised kernel sums to 1), so we skip the adstock
    roll-up here. Saturation/beta live in scaled space; multiply by
    `target_scale` to come back to currency units.
    """
    x_scaled = np.asarray(weekly_spend, dtype=float) / max(channel_scale, 1e-12)
    return beta * _logistic_saturation(x_scaled, lam) * target_scale


def response_curve(
    result: ModelResult,
    channel: str,
    n_points: int = 80,
    max_multiple: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Return (grid, mean_contribution, current_avg_weekly, sat_90_weekly, hdi_low, hdi_high).

    The HDI band is computed by sweeping the grid through every posterior
    draw of (lam, beta) and taking the 3rd / 97th percentiles.
    """
    lam = result.saturation_lam[channel]
    beta = result.betas[channel]
    scale = result.channel_scale[channel]
    target_scale = result.target_scale

    avg_weekly = float(result.spend[channel].mean())
    grid_max = max(avg_weekly, 1.0) * max_multiple
    grid = np.linspace(0.0, grid_max, n_points)

    mean_contrib = _steady_state_contribution(
        grid,
        lam=lam,
        beta=beta,
        channel_scale=scale,
        target_scale=target_scale,
    )

    # Posterior-draw band.
    col = f"{channel}_spend"
    mmm_cache = _cached_mmm()
    if mmm_cache is not None:
        post = mmm_cache.idata.posterior
        lam_draws = post["saturation_lam"].sel(channel=col).to_numpy().ravel()
        beta_draws = post["saturation_beta"].sel(channel=col).to_numpy().ravel()
        # (n_draws, n_grid)
        x_scaled = grid[None, :] / max(scale, 1e-12)
        sat = (1.0 - np.exp(-lam_draws[:, None] * x_scaled)) / (
            1.0 + np.exp(-lam_draws[:, None] * x_scaled)
        )
        contrib_draws = beta_draws[:, None] * sat * target_scale
        hdi_low = np.quantile(contrib_draws, 0.03, axis=0)
        hdi_high = np.quantile(contrib_draws, 0.97, axis=0)
    else:
        hdi_low = mean_contrib.copy()
        hdi_high = mean_contrib.copy()

    # Saturation-90 (weekly spend at which sat func hits 0.9 of its cap=1).
    # logistic_saturation(x, lam) = 0.9  ->  x_scaled = atanh(0.9) / (lam/2)
    # -> x_scaled = ln(19) / lam.
    sat_90_weekly = (np.log(19.0) / max(lam, 1e-9)) * scale

    return grid, mean_contrib, avg_weekly, sat_90_weekly, hdi_low, hdi_high


def optimise_budget(
    result: ModelResult,
    allocation: dict[str, float],
) -> dict[str, float | dict[str, float]]:
    """Predicted annualised revenue for a weekly allocation, using posterior means."""
    weeks = result.n_weeks

    per_channel: dict[str, float] = {}
    for c in result.channels:
        w = float(allocation.get(c, 0.0))
        per_period = float(
            _steady_state_contribution(
                w,
                lam=result.saturation_lam[c],
                beta=result.betas[c],
                channel_scale=result.channel_scale[c],
                target_scale=result.target_scale,
            )
        )
        per_channel[c] = per_period * weeks

    baseline_total = float(result.baseline.mean() * weeks)
    total = baseline_total + sum(per_channel.values())

    roi = {
        c: (per_channel[c] / (allocation.get(c, 0.0) * weeks))
        if allocation.get(c, 0.0) > 0
        else 0.0
        for c in result.channels
    }

    return {
        "total_revenue": float(total),
        "baseline_revenue": baseline_total,
        "channel_contribution": per_channel,
        "channel_roi": roi,
    }


# ---- tiny MMM cache for response_curve HDI sweep ------------------------

_MMM_CACHE: MMM | None = None


def _cached_mmm() -> MMM | None:
    global _MMM_CACHE
    if _MMM_CACHE is None and IDATA_PATH.exists():
        try:
            _MMM_CACHE = MMM.load(str(IDATA_PATH))
        except Exception:  # noqa: BLE001
            _MMM_CACHE = None
    return _MMM_CACHE


def slice_model_result(
    result: ModelResult,
    start: pd.Timestamp | str | None,
    end: pd.Timestamp | str | None,
) -> ModelResult:
    """Restrict all time-aligned series to ``[start, end]`` (inclusive) and recompute metrics."""
    dates = pd.to_datetime(result.dates)
    ts0, ts1 = dates.min(), dates.max()
    t_start = ts0 if start is None else max(pd.Timestamp(start), ts0)
    t_end = ts1 if end is None else min(pd.Timestamp(end), ts1)
    mask = (dates >= t_start) & (dates <= t_end)
    if not bool(mask.any()):
        mask = np.ones(len(dates), dtype=bool)

    def _sl(a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=float)[mask]

    spend = {c: _sl(result.spend[c]) for c in result.channels}
    contributions = {c: _sl(result.contributions[c]) for c in result.channels}
    control_contributions = {k: _sl(v) for k, v in result.control_contributions.items()}
    contribution_hdi = {
        c: (_sl(lo), _sl(hi)) for c, (lo, hi) in result.contribution_hdi.items()
    }

    revenue = _sl(result.revenue)
    fitted = _sl(result.fitted)
    residuals = _sl(result.residuals)
    intercept_contribution = _sl(result.intercept_contribution)
    trend_contribution = _sl(result.trend_contribution)
    seasonality_contribution = _sl(result.seasonality_contribution)
    baseline = _sl(result.baseline)

    dates_out = dates[mask].values

    r2 = _r2(revenue, fitted)
    mape = _mape(revenue, fitted)

    mmm = _cached_mmm()
    if mmm is not None:
        r2_hdi = _r2_hdi_from_components_masked(
            mmm.idata.posterior,
            result.revenue,
            result.target_scale,
            np.asarray(mask),
        )
    else:
        r2_hdi = result.r2_hdi

    ic0 = float(intercept_contribution[0]) if len(intercept_contribution) else result.intercept

    return ModelResult(
        geo=result.geo,
        channels=result.channels,
        dates=dates_out,
        spend=spend,
        revenue=revenue,
        fitted=fitted,
        residuals=residuals,
        contributions=contributions,
        control_contributions=control_contributions,
        intercept_contribution=intercept_contribution,
        trend_contribution=trend_contribution,
        seasonality_contribution=seasonality_contribution,
        baseline=baseline,
        betas=result.betas,
        decays=result.decays,
        saturation_lam=result.saturation_lam,
        channel_scale=result.channel_scale,
        target_scale=result.target_scale,
        intercept=ic0,
        r2=r2,
        mape=mape,
        r2_hdi=r2_hdi,
        contribution_hdi=contribution_hdi,
    )


def posterior_predictive_interval_df(result: ModelResult) -> pd.DataFrame | None:
    """94% predictive intervals for revenue (original scale) from cached InferenceData."""
    mmm = _cached_mmm()
    if mmm is None:
        return None
    return mmm.summary.posterior_predictive(hdi_probs=[0.94])


def posterior_predictive_interval_for_result(result: ModelResult) -> pd.DataFrame | None:
    """Posterior predictive summary rows aligned to ``result.dates`` (for filtered windows)."""
    df = posterior_predictive_interval_df(result)
    if df is None:
        return None
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    want = pd.to_datetime(result.dates)
    out = out[out["date"].isin(want)].sort_values("date")
    return out


def paid_increment_hdi_arrays(result: ModelResult) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-date 94% HDI for total paid media contribution (posterior sum over channels)."""
    mmm = _cached_mmm()
    if mmm is None:
        return None
    post = mmm.idata.posterior
    ts = result.target_scale
    paid = post["channel_contribution"].sum("channel") * ts
    arr_full = paid.stack(sample=("chain", "draw")).transpose("date", "sample").to_numpy()
    alpha = 0.03
    low_full = np.quantile(arr_full, alpha, axis=1)
    high_full = np.quantile(arr_full, 1 - alpha, axis=1)
    all_dates = pd.to_datetime(paid.coords["date"].values)
    sel = pd.to_datetime(result.dates)
    pos = all_dates.get_indexer(sel)
    if (pos < 0).any():
        return None
    return low_full[pos], high_full[pos]


def marginal_slope_roas(result: ModelResult, channel: str) -> float:
    """Slope of incremental weekly revenue vs weekly spend at mean spend (ROAS factor)."""
    w0 = float(np.mean(result.spend[channel]))
    if w0 <= 0:
        return 0.0
    eps = max(1.0, w0 * 0.001)
    lam = result.saturation_lam[channel]
    beta = result.betas[channel]
    scale = result.channel_scale[channel]
    ts = result.target_scale

    def g(w: float) -> float:
        return float(
            _steady_state_contribution(
                w,
                lam=lam,
                beta=beta,
                channel_scale=scale,
                target_scale=ts,
            )
        )

    return (g(w0 + eps) - g(w0 - eps)) / (2 * eps)


def channel_window_roi_hdi(
    result: ModelResult, channel: str
) -> tuple[float, float, float]:
    """Point mean ROI and 94% HDI (posterior contribution sum / observed spend)."""
    mmm = _cached_mmm()
    col = f"{channel}_spend"
    spend_total = result.total_spend[channel]
    if spend_total <= 0:
        return 0.0, 0.0, 0.0
    if mmm is None:
        roi = result.total_contribution[channel] / spend_total
        return roi, roi, roi
    post = mmm.idata.posterior
    ts = result.target_scale
    da = post["channel_contribution"].sel(channel=col)
    all_dates = pd.to_datetime(da.coords["date"].values)
    sel = pd.to_datetime(result.dates)
    pos = all_dates.get_indexer(sel)
    if (pos < 0).any():
        roi = result.total_contribution[channel] / spend_total
        return roi, roi, roi
    da = da.isel(date=pos)
    draws = da.sum("date").stack(sample=("chain", "draw")).to_numpy() * ts
    roi_s = draws / spend_total
    return (
        float(np.mean(roi_s)),
        float(np.quantile(roi_s, 0.03)),
        float(np.quantile(roi_s, 0.97)),
    )


def recommended_weekly_allocation(result: ModelResult) -> dict[str, float]:
    """Approximately optimal weekly mix under steady-state `optimise_budget` (SLSQP)."""
    from scipy.optimize import minimize

    channels = result.channels
    cur = {c: float(np.mean(result.spend[c])) for c in channels}
    total = float(sum(cur.values())) or 1.0

    def neg_rev(w_raw: np.ndarray) -> float:
        w = np.maximum(w_raw, 0.0)
        s = float(w.sum())
        if s <= 0:
            return 0.0
        w = w / s
        alloc = {c: w[i] * total for i, c in enumerate(channels)}
        return -float(optimise_budget(result, alloc)["total_revenue"])

    n = len(channels)
    x0 = np.array([cur[c] / total for c in channels])
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0},)
    res = minimize(
        neg_rev,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 800, "ftol": 1e-6},
    )
    w = np.maximum(res.x, 0.0)
    w = w / w.sum()
    return {c: float(w[i] * total) for i, c in enumerate(channels)}


# Re-export for callers that used to import adstock/hill from here.
def adstock(x: np.ndarray, decay: float) -> np.ndarray:  # pragma: no cover - compat
    """Normalised geometric adstock (convolution with alpha^lag weights)."""
    x = np.asarray(x, dtype=float)
    l_max = ADSTOCK_L_MAX
    weights = np.power(decay, np.arange(l_max))
    weights = weights / weights.sum()
    out = np.zeros_like(x)
    for i in range(len(x)):
        for lag, w in enumerate(weights):
            if i - lag >= 0:
                out[i] += w * x[i - lag]
    return out


def hill(x: np.ndarray | float, lam: float, _s: float | None = None) -> np.ndarray | float:  # pragma: no cover - compat
    """Logistic saturation — compatibility alias for old callers."""
    return _logistic_saturation(x, lam)

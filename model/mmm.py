"""Bayesian MMM backed by `pymc-marketing`.

Architecture:
    Revenue_t = intercept
              + linear_trend_t              (via the `t` control column)
              + yearly_seasonality_t        (Fourier, n_order=3)
              + sum_k gamma_k * control_k_t (economic controls)
              + sum_c beta_c * LogisticSaturation(
                        GeometricAdstock(spend_{c,t}, alpha_c, l_max=ADSTOCK_L_MAX),
                        lam_c,
                    )

The heavy lifting is done by `pymc_marketing.mmm.MMM` which samples with NUTS.
The fitted `InferenceData` is cached to `data/mmm_idata.nc` so only the first
launch pays the sampling cost. NUTS ``draws`` / ``tune`` / ``target_accept`` are
read from ``data/mmm_sampler_config.json`` when present (see dashboard Options)
and are folded into the cache fingerprint alongside ``ADSTOCK_L_MAX``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from data.loader import CHANNELS, CONTROLS, TREND_COLUMNS

from .sampling_progress import SamplingProgressTracker

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

# macOS: clang++ often fails on `#include <vector>` (SDK / toolchain mismatch). Disabling
# PyTensor's C++ compiler forces pure NumPy/C fallbacks; pair with nutpie for NUTS.
# Set PYTENSOR_CXX to a working compiler path to re-enable C++ ops instead.
if sys.platform == "darwin":
    import pytensor

    pytensor.config.cxx = os.environ.get("PYTENSOR_CXX", "")

from pymc_marketing.mmm import (
    GeometricAdstock,
    LogisticSaturation,
)
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.prior import Prior

# ---- constants ----------------------------------------------------------

CHANNEL_COLUMNS: list[str] = [f"{c}_spend" for c in CHANNELS]
CONTROL_COLUMNS: list[str] = CONTROLS + TREND_COLUMNS
# Single geometric-adstock lag horizon (weeks) for all channels — not tied to labels.
ADSTOCK_L_MAX = 12
FOURIER_N_ORDER = 3

# ``MODEL_CONFIG_VERSION`` is folded into the idata fingerprint so any change
# to the prior structure forces a refit of the cached posterior.
MODEL_CONFIG_VERSION = "v3-gamma-saturation-priors-2026-04"

IDATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mmm_idata.nc"
IDATA_FINGERPRINT_PATH = Path(__file__).resolve().parent.parent / "data" / "mmm_idata.sha256"
SAMPLER_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "mmm_sampler_config.json"

DEFAULT_SAMPLER_CONFIG: dict[str, float | int] = {
    "draws": 2000,
    "tune": 2000,
    "target_accept": 0.95,
}


def load_sampler_config() -> dict[str, float | int]:
    """Merge ``data/mmm_sampler_config.json`` with defaults (dashboard Options)."""
    out = dict(DEFAULT_SAMPLER_CONFIG)
    if not SAMPLER_CONFIG_PATH.exists():
        return out
    try:
        raw = json.loads(SAMPLER_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    if isinstance(raw, dict):
        if "draws" in raw:
            out["draws"] = max(100, int(raw["draws"]))
        if "tune" in raw:
            out["tune"] = max(200, int(raw["tune"]))
        if "target_accept" in raw:
            ta = float(raw["target_accept"])
            out["target_accept"] = min(0.9999, max(0.75, ta))
    return out


def save_sampler_config(cfg: dict[str, float | int]) -> None:
    """Persist sampler settings for the next app launch."""
    SAMPLER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLER_CONFIG_PATH.write_text(
        json.dumps(
            {
                "draws": int(cfg["draws"]),
                "tune": int(cfg["tune"]),
                "target_accept": float(cfg["target_accept"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _idata_fingerprint(
    df: pd.DataFrame, sampler: dict[str, float | int] | None = None
) -> str:
    """Stable hash of training data + model spec + NUTS settings."""
    s = sampler if sampler is not None else load_sampler_config()
    h = hashlib.sha256()
    h.update("|".join(CHANNEL_COLUMNS).encode("utf-8"))
    h.update(b"||")
    h.update("|".join(CONTROL_COLUMNS).encode("utf-8"))
    h.update(
        f"||{ADSTOCK_L_MAX}||{FOURIER_N_ORDER}||{MODEL_CONFIG_VERSION}".encode("utf-8")
    )
    h.update(
        f"||draws={int(s['draws'])}||tune={int(s['tune'])}||"
        f"ta={float(s['target_accept']):.6f}".encode("utf-8")
    )
    cols = ["time", *CHANNEL_COLUMNS, *CONTROL_COLUMNS, "revenue"]
    sub = df[cols].sort_values("time").reset_index(drop=True)
    h.update(pd.util.hash_pandas_object(sub, index=True).values.tobytes())
    return h.hexdigest()


def _log_mcmc_diagnostics(mmm: MMM) -> None:
    """Log divergences, R-hat, and ESS for key parameters after sampling."""
    idata = mmm.idata
    div = 0
    try:
        div = int(idata.sample_stats["diverging"].sum().item())
    except (KeyError, AttributeError):
        pass
    print(f"[MMM] MCMC divergences: {div}", flush=True)
    if div > 0:
        warnings.warn(
            f"NUTS reported {div} divergences, consider more tuning or reparameterisation.",
            RuntimeWarning,
            stacklevel=2,
        )
    if os.environ.get("MMM_STRICT_DIAGNOSTICS") == "1" and div > 0:
        raise RuntimeError(
            f"MMM_STRICT_DIAGNOSTICS is set: refusing to continue with {div} divergences"
        )

    post = idata.posterior
    var_names = [
        v
        for v in (
            "intercept",
            "adstock_alpha",
            "saturation_lam",
            "saturation_beta",
        )
        if v in post
    ]
    if not var_names:
        return
    summ = az.summary(idata, var_names=var_names, round_to=4)
    print("[MMM] ArviZ summary (key parameters):\n" + summ.to_string(), flush=True)
    rhat_max = float(summ["r_hat"].max())
    ess_bulk_min = float(summ["ess_bulk"].min()) if "ess_bulk" in summ.columns else float("nan")
    print(
        f"[MMM] max r_hat={rhat_max:.4f}, min ess_bulk={ess_bulk_min:.1f}",
        flush=True,
    )
    if rhat_max > 1.01:
        warnings.warn(
            f"max r_hat {rhat_max:.4f} exceeds 1.01; chains may not have mixed.",
            UserWarning,
            stacklevel=2,
        )


def implied_half_life_weeks(alpha: float) -> float:
    """Weeks until a one-off pulse retains 50% of its geometric adstock weight.

    For a carryover coefficient ``alpha`` in (0, 1): half-life = ln(0.5) / ln(alpha).
    """
    a = float(np.clip(alpha, 1e-6, 1.0 - 1e-9))
    return float(np.log(0.5) / np.log(a))


def mcmc_diagnostics_bundle(idata) -> dict[str, float | int | list | None]:
    """Compact MCMC stats for the dashboard (R-hat, ESS, divergences, energy, BFMI)."""
    out: dict[str, float | int | list | None] = {}
    try:
        out["divergences"] = int(idata.sample_stats["diverging"].sum().item())
    except (KeyError, AttributeError, TypeError):
        out["divergences"] = None
    try:
        post_vars = [str(v) for v in idata.posterior.data_vars.keys()]
        summ = az.summary(idata, var_names=post_vars, round_to=4)
        out["max_r_hat"] = float(summ["r_hat"].max())
        out["min_ess_bulk"] = float(summ["ess_bulk"].min())
        out["min_ess_tail"] = float(summ["ess_tail"].min())
    except Exception:  # noqa: BLE001
        out["max_r_hat"] = None
        out["min_ess_bulk"] = None
        out["min_ess_tail"] = None
    try:
        bfmi = az.bfmi(idata)
        arr = np.asarray(bfmi).ravel()
        out["bfmi_per_chain"] = [float(x) for x in arr]
        out["bfmi_mean"] = float(np.mean(arr))
    except Exception:  # noqa: BLE001
        out["bfmi_per_chain"] = None
        out["bfmi_mean"] = None
    try:
        en = idata.sample_stats["energy"]
        out["energy_mean"] = float(en.mean().item())
        out["energy_sd"] = float(en.std().item())
    except (KeyError, AttributeError, TypeError):
        out["energy_mean"] = None
        out["energy_sd"] = None
    try:
        out["chains"] = int(idata.posterior.sizes["chain"])
        out["draws_per_chain"] = int(idata.posterior.sizes["draw"])
    except (KeyError, AttributeError, TypeError):
        out["chains"] = None
        out["draws_per_chain"] = None
    return out


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

    # Inference / UI metadata (optional for backwards compatibility)
    mcmc_diagnostics: dict[str, float | int | list | None] | None = None
    sampler_config: dict[str, float | int] | None = None
    adstock_l_max: int = ADSTOCK_L_MAX
    half_life_truncation_warning: str | None = None

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


def _informed_model_config(df: pd.DataFrame) -> dict:
    """Build a ``model_config`` with priors calibrated to the training data.

    * ``saturation_beta`` uses ``Gamma(alpha=3, beta=3/beta_mean)`` per channel.
      The prior mean is still anchored to each channel's share of total spend
      (high-spend channels get a wider prior), but the mode is at
      ``2·beta_mean/3`` (strictly positive rather than zero).
    * ``saturation_lam`` is tightened from ``Gamma(3, 1)`` (mean 3, heavy
      left tail) to ``Gamma(4, 2)`` (mean 2, mode 1.5) to concentrate
      diminishing-returns curvature in the identifiable region.
    * ``intercept`` and ``gamma_control`` stay tight around scaled-revenue
      magnitudes so the sampler doesn't waste mass on impossible regions.
    * ``adstock_alpha`` remains ``Beta(3, 3)`` — symmetric around 0.5, so
      the decay parameter no longer pushed toward 0 by the default
      ``Beta(1, 3)`` whose mean is 0.25.
    """
    spend_totals = np.array(
        [float(df[col].sum()) for col in CHANNEL_COLUMNS], dtype=float
    )
    total = spend_totals.sum()
    if total <= 0:
        spend_shares = np.full(len(CHANNEL_COLUMNS), 1.0 / len(CHANNEL_COLUMNS))
    else:
        spend_shares = spend_totals / total
    # Floor so minor channels still get non-trivial prior mass; ceiling avoids
    # arbitrarily fat tails for the single dominant channel. Wrap in a
    # DataArray with explicit ``channel`` dim so ``pymc_extras.Prior`` does
    # not emit an implicit-conversion warning.
    beta_mean = xr.DataArray(
        np.clip(spend_shares, 0.05, 0.5),
        dims=("channel",),
        coords={"channel": list(CHANNEL_COLUMNS)},
    )

    return {
        "intercept": Prior("Normal", mu=0.35, sigma=0.15, dims=()),
        "likelihood": Prior(
            "Normal",
            sigma=Prior("HalfNormal", sigma=0.15, dims=()),
            dims="date",
        ),
        "gamma_control": Prior("Normal", mu=0.0, sigma=0.1, dims="control"),
        "gamma_fourier": Prior("Laplace", mu=0.0, b=0.05, dims="fourier_mode"),
        "adstock_alpha": Prior("Beta", alpha=3.0, beta=3.0, dims="channel"),
        "saturation_lam": Prior("Gamma", alpha=4.0, beta=2.0, dims="channel"),
        "saturation_beta": Prior(
            "Gamma", alpha=3.0, beta=3.0 / beta_mean, dims="channel"
        ),
    }


def _build_mmm(df: pd.DataFrame) -> MMM:
    return MMM(
        date_column="time",
        channel_columns=CHANNEL_COLUMNS,
        target_column="revenue",
        control_columns=CONTROL_COLUMNS,
        adstock=GeometricAdstock(l_max=ADSTOCK_L_MAX),
        saturation=LogisticSaturation(),
        yearly_seasonality=FOURIER_N_ORDER,
        model_config=_informed_model_config(df),
    )


def _fit_mmm(
    df: pd.DataFrame,
    sampler: dict[str, float | int],
    progress: SamplingProgressTracker | None = None,
) -> MMM:
    mmm = _build_mmm(df)
    X = df[["time", *CHANNEL_COLUMNS, *CONTROL_COLUMNS]]
    y = df["revenue"]
    chains = 4
    draws = int(sampler["draws"])
    tune = int(sampler["tune"])
    fit_kw: dict = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "target_accept": float(sampler["target_accept"]),
        "random_seed": 10,
        "progressbar": True,
        # Always nutpie+JAX: native PyMC NUTS can raise "zero to a negative power" in
        # pymc_marketing's geometric adstock (`alpha ** lag` at alpha=0, lag=0). Nutpie
        # does not hit that PyTensor path. Per-draw `callback=` exists only on pymc NUTS.
        "nuts_sampler": "nutpie",
        "nuts_sampler_kwargs": {"backend": "jax"},
    }
    if progress is not None:
        progress.reset(
            chains=chains, tune=tune, draws=draws, indeterminate=True
        )
    mmm.fit(X, y, **fit_kw)
    if progress is not None:
        progress.set_phase("ppc")
    mmm.sample_posterior_predictive(X, extend_idata=True, progressbar=True)
    if progress is not None:
        progress.set_phase("idle")
    _log_mcmc_diagnostics(mmm)
    return mmm


def _invalidate_idata_cache() -> None:
    for path in (IDATA_PATH, IDATA_FINGERPRINT_PATH):
        try:
            path.unlink()
        except OSError:
            pass


def _reset_mmm_cache() -> None:
    global _MMM_CACHE
    _MMM_CACHE = None


def _load_or_fit(
    df: pd.DataFrame,
    sampler: dict[str, float | int],
    progress: SamplingProgressTracker | None = None,
) -> MMM:
    fp = _idata_fingerprint(df, sampler)
    cache_ok = (
        IDATA_PATH.exists()
        and IDATA_FINGERPRINT_PATH.exists()
        and IDATA_FINGERPRINT_PATH.read_text(encoding="utf-8").strip() == fp
    )
    if cache_ok:
        try:
            return MMM.load(str(IDATA_PATH))
        except Exception:  # noqa: BLE001
            _invalidate_idata_cache()
    elif IDATA_PATH.exists() and not cache_ok:
        # Stale or legacy cache (missing/wrong fingerprint).
        print(
            "[MMM] Inference cache miss (data or model spec changed); refitting...",
            flush=True,
        )
        _invalidate_idata_cache()

    print(
        "Fitting pymc-marketing MMM (first run ~60s, cached after)...",
        flush=True,
    )
    mmm = _fit_mmm(df, sampler, progress=progress)
    try:
        IDATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        mmm.save(str(IDATA_PATH))
        IDATA_FINGERPRINT_PATH.write_text(fp + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"(warning) failed to cache InferenceData: {exc}", flush=True)
    _reset_mmm_cache()
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


def _build_result(
    mmm: MMM,
    df: pd.DataFrame,
    geo_label: str,
    sampler: dict[str, float | int],
) -> ModelResult:
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

    if TREND_COLUMNS:
        trend_col = TREND_COLUMNS[0]
        trend_contribution = (
            control_contribution_scaled.sel(control=trend_col).to_numpy() * target_scale
        )
    else:
        trend_contribution = np.zeros(len(df))

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

    mcmc_diag = mcmc_diagnostics_bundle(mmm.idata)
    half_life_by_ch: dict[str, float] = {
        c: implied_half_life_weeks(decays_out[c]) for c in CHANNELS
    }

    trunc_note: str | None = None
    over = [c for c, hl in half_life_by_ch.items() if hl > float(ADSTOCK_L_MAX) + 0.5]
    if over:
        trunc_note = (
            f"Posterior mean half-life exceeds adstock l_max={ADSTOCK_L_MAX} weeks for: "
            f"{', '.join(over)}. The geometric kernel is truncated; consider raising "
            f"ADSTOCK_L_MAX in model/mmm.py."
        )

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
        mcmc_diagnostics=mcmc_diag,
        sampler_config=dict(sampler),
        adstock_l_max=ADSTOCK_L_MAX,
        half_life_truncation_warning=trunc_note,
    )


def fit_surrogate(
    df: pd.DataFrame,
    geo_label: str = "All",
    sampler_config: dict[str, float | int] | None = None,
    *,
    progress: SamplingProgressTracker | None = None,
) -> ModelResult:
    """Fit (or load from disk cache) the Bayesian MMM and assemble ModelResult."""
    sampler = (
        dict(sampler_config) if sampler_config is not None else load_sampler_config()
    )
    mmm = _load_or_fit(df, sampler, progress=progress)
    return _build_result(mmm, df, geo_label, sampler)


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
    """Predicted total revenue over the fit window for a weekly allocation (posterior means).

    Scales steady-state weekly contributions by ``result.n_weeks``; not calendar-year annualisation.
    """
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
        mcmc_diagnostics=result.mcmc_diagnostics,
        sampler_config=result.sampler_config,
        adstock_l_max=result.adstock_l_max,
        half_life_truncation_warning=result.half_life_truncation_warning,
    )


def posterior_predictive_interval_df(result: ModelResult) -> pd.DataFrame | None:
    """94% predictive intervals for revenue (original scale) from cached InferenceData.

    Bypasses ``mmm.summary.posterior_predictive``: in the single-geo multidimensional
    model path, pymc-marketing stores ``target_scale`` with a spurious length-1
    ``target_scale_dim_0`` coordinate that leaks into ``custom_dims`` and makes the
    library's internal merge with ``observed`` raise ``KeyError``. We compute the HDI
    directly from ``idata.posterior_predictive["y"]`` using the known scalar
    ``result.target_scale`` to stay in original currency units.
    """
    mmm = _cached_mmm()
    if mmm is None:
        return None
    pp = getattr(mmm.idata, "posterior_predictive", None)
    if pp is None or "y" not in pp:
        return None

    y_scaled = pp["y"]
    keep = {"chain", "draw", "date"}
    squeezable = [
        d for d in y_scaled.dims if d not in keep and y_scaled.sizes[d] == 1
    ]
    if squeezable:
        y_scaled = y_scaled.isel({d: 0 for d in squeezable}, drop=True)

    y = y_scaled * float(result.target_scale)
    hdi = az.hdi(y, hdi_prob=0.94)["y"]
    lower = hdi.sel(hdi="lower").to_numpy().astype(float)
    upper = hdi.sel(hdi="higher").to_numpy().astype(float)
    dates = pd.to_datetime(y.coords["date"].values)
    return pd.DataFrame(
        {
            "date": dates,
            "abs_error_94_lower": lower,
            "abs_error_94_upper": upper,
        }
    )


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
    """Approximately optimal weekly mix under steady-state `optimise_budget` (SLSQP).

    Optimises only the paid-media contribution (the baseline is constant w.r.t. the
    allocation) and supplies an analytic Jacobian so SLSQP's finite-difference
    gradient doesn't get swamped by the large baseline revenue term — which
    previously caused ``ftol`` to fire immediately and return the current mix.
    """
    from scipy.optimize import minimize

    channels = result.channels
    cur = {c: float(np.mean(result.spend[c])) for c in channels}
    total = float(sum(cur.values())) or 1.0

    lam = np.array([float(result.saturation_lam[c]) for c in channels])
    beta = np.array([float(result.betas[c]) for c in channels])
    scale = np.array(
        [max(float(result.channel_scale[c]), 1e-12) for c in channels]
    )

    def _sat(x: np.ndarray) -> np.ndarray:
        return (1.0 - np.exp(-lam * x)) / (1.0 + np.exp(-lam * x))

    def neg_paid(w: np.ndarray) -> float:
        spend = np.maximum(w, 0.0) * total
        sat = _sat(spend / scale)
        return -float(np.sum(beta * sat))

    def jac(w: np.ndarray) -> np.ndarray:
        spend = np.maximum(w, 0.0) * total
        sat = _sat(spend / scale)
        d_contrib_dw = beta * (lam * (1.0 - sat * sat) / 2.0) * (total / scale)
        return -d_contrib_dw

    n = len(channels)
    x0 = np.array([cur[c] / total for c in channels])
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0},)
    res = minimize(
        neg_paid,
        x0,
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 800, "ftol": 1e-10},
    )
    w = np.maximum(res.x, 0.0)
    s = float(w.sum())
    if s <= 0:
        return {c: cur[c] for c in channels}
    w = w / s
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

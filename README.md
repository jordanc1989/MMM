# MMM Dashboard

A local web dashboard for **Bayesian Media Mix Modeling (MMM)**. It fits a hierarchical MMM on sample marketing data, then explores channel effects, budget trade-offs and optimisation in a semi-interactive UI.

## What it does

The app uses [PyMC-Marketing](https://www.pymc-marketing.io/) to estimate a multidimensional MMM with **geometric adstock** and **logistic saturation** on paid-media spend, plus controls and seasonality. Inference is **NUTS** (Hamiltonian Monte Carlo), posteriors are summarised with [ArviZ](https://python.arviz.org/) / [xarray](https://docs.xarray.dev/).

**Data:** On first run it downloads Google [Meridian](https://github.com/google/meridian)’s simulated `geo_all_channels.csv` and caches it under `data/`. This is synthetic multi-geo weekly data—not your production numbers—intended for demos and development.

**Caching:** Fitted `InferenceData` is written to `data/mmm_idata.nc` (and a fingerprint file) so later launches reload the posterior instead of resampling, unless you refit or invalidate the cache.

## Pages

| Route | Purpose |
|--------|---------|
| **Overview** | KPIs, fit diagnostics, revenue vs. baseline/media decomp over time |
| **Contributions** | Channel contribution to revenue (posterior uncertainty) |
| **Response curves** | Marginal response / saturation curves by channel |
| **Optimiser** | Budget scenarios and recommended channel allocation derived from the fitted model |

The header **Options** panel lets you adjust sampler settings (draws, tuning steps, target accept) and trigger a **refit**; successful runs persist settings to `data/mmm_sampler_config.json`.

## Stack

- **UI:** [Dash](https://dash.plotly.com/) 4.x, [Dash Mantine Components](https://www.dash-mantine-components.com/), [Plotly](https://plotly.com/python/) figures  
- **Model:** `pymc-marketing`, PyMC, optional **nutpie** + **JAX** for sampling throughput  
- **Python:** 3.10+ recommended (match your `pymc-marketing` wheel availability)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8050**. The first model fit can take on the order of a minute while NUTS runs. Subsequent starts are faster when the NetCDF cache is present.

## Configuration & generated files

- `data/mmm_sampler_config.json` — optional; merged with built-in defaults for `draws`, `tune`, `target_accept`  
- `data/mmm_idata.nc` — cached posterior (gitignored by default)  
- `.pytensor_cache/` — PyTensor compile cache (gitignored)

## macOS note

PyMC/PyTensor may need a working C++ toolchain or fall back to pure NumPy modes. The project sets a local PyTensor compile dir and, on Apple systems, can disable the C++ compiler unless `PYTENSOR_CXX` points to a working compiler. If sampling fails, check PyMC/PyTensor docs for your OS version.

## Disclaimer

This repository is a **demonstration** built on **simulated** data. Do not treat its outputs as business decisions without validating on your own data, priors, and governance.

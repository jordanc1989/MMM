# syntax=docker/dockerfile:1
# Hugging Face Spaces (Docker SDK) image for the Bayesian MMM Dash dashboard.
FROM python:3.13-slim

# build-essential / g++: pytensor compiles the PyMC model graph to C at runtime
# when the cached posterior is loaded, so a C++ toolchain must be present.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, fully-locked dependency installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# HF Spaces runs containers as a non-root user with uid 1000; everything the app
# writes at runtime must live under this user's home.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    # Redirect every runtime cache the scientific stack writes to a writable home.
    PYTENSOR_FLAGS=base_compiledir=/home/user/.pytensor \
    MPLCONFIGDIR=/home/user/.config/matplotlib \
    NUMBA_CACHE_DIR=/home/user/.cache/numba \
    XDG_CACHE_HOME=/home/user/.cache

WORKDIR /home/user/app

# Install dependencies first so this layer is cached unless the lockfile changes.
# README.md is referenced by pyproject metadata, so it must be present too.
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# Copy the rest of the application (includes the pre-fitted data/mmm_idata.nc
# cache, so startup loads the posterior instead of resampling).
COPY --chown=user . .

EXPOSE 7860

# One worker: the model cache is an in-process global loaded once at boot, and a
# single 25 MB posterior is plenty for a demo. --preload loads it in the master
# before forking; --timeout 240 covers the heavier on-demand response-curve math.
CMD ["uv", "run", "--no-sync", "gunicorn", "wsgi:server", \
     "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", \
     "--preload", "--timeout", "240", "--pythonpath", "/home/user/app"]

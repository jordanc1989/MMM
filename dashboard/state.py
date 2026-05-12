"""Shared runtime state for Dash Pages layout functions."""

from __future__ import annotations

from model.mmm import ModelResult

_model_cache: dict[str, ModelResult] | None = None


def set_model_cache(cache: dict[str, ModelResult]) -> None:
    global _model_cache
    _model_cache = cache


def current_result() -> ModelResult:
    if _model_cache is None:
        raise RuntimeError("Model cache has not been initialised.")
    return _model_cache["All"]

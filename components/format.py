"""Shared number formatting so values read identically on every page."""

from __future__ import annotations


def fmt_currency(v: float) -> str:
    """Compact USD: billions/millions to 2dp, thousands to 1dp, else whole dollars."""
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.1f}K"
    return f"${v:,.0f}"


def fmt_pct(x: float | int | None, *, dp: int = 1) -> str:
    """Format a 0–1 fraction as a percentage, or an em dash for ``None``."""
    if x is None:
        return "—"
    return f"{float(x) * 100:.{dp}f}%"

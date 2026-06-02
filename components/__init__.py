"""Shared UI components and chart theming."""
from .chart_theme import (
    ACCENT,
    CHANNEL_COLORS,
    CHART_FONT_COLOR,
    DATA_INK,
    GRID_COLOR,
    NEGATIVE_COLOR,
    PALETTE,
    POSITIVE_COLOR,
    STEEL_COLOR,
    ZERO_LINE_COLOR,
    apply_dark_theme,
    with_alpha,
)
from .format import fmt_currency, fmt_pct
from .kpi_card import kpi_card
from .layout import section, page_header

__all__ = [
    "apply_dark_theme",
    "with_alpha",
    "ACCENT",
    "CHANNEL_COLORS",
    "CHART_FONT_COLOR",
    "DATA_INK",
    "GRID_COLOR",
    "NEGATIVE_COLOR",
    "PALETTE",
    "POSITIVE_COLOR",
    "STEEL_COLOR",
    "ZERO_LINE_COLOR",
    "fmt_currency",
    "fmt_pct",
    "kpi_card",
    "section",
    "page_header",
]

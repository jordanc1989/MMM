"""Shared UI components and chart theming."""
from .chart_theme import (
    CHANNEL_COLORS,
    CHART_FONT_COLOR,
    PALETTE,
    apply_dark_theme,
)
from .kpi_card import kpi_card
from .layout import section, page_header

__all__ = [
    "apply_dark_theme",
    "CHANNEL_COLORS",
    "CHART_FONT_COLOR",
    "PALETTE",
    "kpi_card",
    "section",
    "page_header",
]

"""Shared Dash component ids for cross-page callbacks."""

# Session-persisted analysis window for Model Overview only (other pages use full period).
OVERVIEW_DATE_STORE = "overview-date-store"
OVERVIEW_RANGE_PRESET = "overview-range-preset"
OVERVIEW_YEAR_SELECT = "overview-year-select"
OVERVIEW_TOOLBAR = "overview-toolbar"

# Global model / NUTS options (header) + bump to refresh all pages after refit.
MODEL_REFRESH_STORE = "model-refresh-store"
OPT_TARGET_ACCEPT = "mmm-opt-target-accept"
OPT_DRAWS = "mmm-opt-draws"
OPT_TUNE = "mmm-opt-tune"
OPT_REFIT_BTN = "mmm-opt-refit-btn"
OPT_REFIT_STATUS = "mmm-opt-refit-status"
REFIT_OVERLAY_STORE = "mmm-refit-overlay-store"
REFIT_OVERLAY_ROOT = "mmm-refit-overlay-root"
REFIT_POLL_INTERVAL = "mmm-refit-poll-interval"
REFIT_PROGRESS_CHAINS = "mmm-refit-progress-chains"
REFIT_JOB_STORE = "mmm-refit-job-store"

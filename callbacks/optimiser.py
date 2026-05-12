"""Callbacks for the budget optimiser page."""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from components.ids import MODEL_REFRESH_STORE
from figures.optimiser import _allocation_donut
from layouts.optimiser import (
    ALLOC_LABEL_ID,
    ALLOC_PROPOSED_GRAPH_ID,
    ALLOC_RECOMMENDED_GRAPH_ID,
    APPLY_MODEL_MIX_ID,
    CONSTRAINT_MAX_ID,
    CONSTRAINT_MIN_ID,
    OPTIMISER_STATUS_ID,
    PREDICTED_HDI_ID,
    PREDICTED_REV_ID,
    RESET_BUTTON_ID,
    ROI_TABLE_ID,
    SLIDER_ID,
    UPLIFT_ID,
    UPLIFT_PROB_ID,
    UTILISATION_ID,
    WEIGHT_LABEL_ID,
    _constraint_dict,
    _current_weekly_alloc,
    _default_weights,
    _fmt_currency,
    _fmt_pct,
    _fmt_weight,
    _outside_support_count,
    _roi_rows,
    _roi_table,
    _weights_from_weekly_alloc,
)
from model.mmm import (
    ModelResult,
    optimise_budget,
    posterior_budget_summary,
    recommended_weekly_allocation,
    steady_state_current_budget_prediction,
)

def register_optimiser_callbacks(app, results_by_geo: dict[str, ModelResult]) -> None:
    @app.callback(
        Output(PREDICTED_REV_ID, "children"),
        Output(PREDICTED_HDI_ID, "children"),
        Output(UPLIFT_ID, "children"),
        Output(UPLIFT_ID, "c"),
        Output(UTILISATION_ID, "children"),
        Output(UPLIFT_PROB_ID, "children"),
        Output(ROI_TABLE_ID, "children"),
        Output(ALLOC_PROPOSED_GRAPH_ID, "figure"),
        Output({**WEIGHT_LABEL_ID, "channel": ALL}, "children"),
        Output({**ALLOC_LABEL_ID, "channel": ALL}, "children"),
        Input({**SLIDER_ID, "channel": ALL}, "value"),
        Input(MODEL_REFRESH_STORE, "data"),
        State({**SLIDER_ID, "channel": ALL}, "id"),
    )
    def _recalc(values, _refresh, ids):
        result = results_by_geo["All"]
        channels = [i["channel"] for i in ids]

        raw = [float(v or 0) for v in values]
        total_w = sum(raw) or 1.0
        weights_pct = [v / total_w for v in raw]

        current_alloc = _current_weekly_alloc(result)
        total_weekly = sum(current_alloc.values())

        new_alloc = {c: weights_pct[i] * total_weekly for i, c in enumerate(channels)}

        current_pred = steady_state_current_budget_prediction(result)
        new_pred = optimise_budget(result, new_alloc)
        current_summary = posterior_budget_summary(
            result,
            current_alloc,
            current_allocation=current_alloc,
        )
        posterior = posterior_budget_summary(
            result,
            new_alloc,
            current_allocation=current_alloc,
        )

        cur_rev = float(current_summary["expected_revenue"])
        new_rev = float(posterior["expected_revenue"])
        uplift = new_rev - cur_rev
        uplift_color = "teal" if uplift >= 0 else "red"
        uplift_str = ("+" if uplift >= 0 else "") + _fmt_currency(uplift)
        pct_change = (uplift / cur_rev * 100) if cur_rev else 0.0
        outside = _outside_support_count(result, new_alloc)
        support_note = (
            f"; {outside} channel{'s' if outside != 1 else ''} outside observed support"
            if outside
            else "; all channels inside observed support"
        )
        utilisation = f"{pct_change:+.2f}% vs current allocation{support_note}"
        hdi_low, hdi_high = posterior["revenue_hdi"]  # type: ignore[assignment]
        threshold = float(posterior["uplift_threshold"])
        prob_line = (
            f"P(proposed > current): {_fmt_pct(float(posterior['prob_proposed_gt_current']))} · "
            f"P(uplift > {_fmt_currency(threshold)}): "
            f"{_fmt_pct(float(posterior['prob_uplift_gt_threshold']))}"
        )

        rows = _roi_rows(
            result,
            channels,
            current_alloc,
            new_alloc,
            current_pred,
            new_pred,
            result.n_weeks,
        )

        prop_fig = _allocation_donut(
            channels, new_alloc, title="Proposed (sliders)"
        )

        weight_children = [_fmt_weight(w * 100.0) for w in weights_pct]
        alloc_children = [_fmt_currency(new_alloc[c]) + " / week" for c in channels]

        return (
            _fmt_currency(new_rev),
            f"94% HDI {_fmt_currency(float(hdi_low))}–{_fmt_currency(float(hdi_high))}",
            uplift_str,
            uplift_color,
            utilisation,
            prob_line,
            _roi_table(rows),
            prop_fig,
            weight_children,
            alloc_children,
        )

    @app.callback(
        Output({**SLIDER_ID, "channel": ALL}, "value"),
        Output(ALLOC_RECOMMENDED_GRAPH_ID, "figure"),
        Output(OPTIMISER_STATUS_ID, "children"),
        Input(RESET_BUTTON_ID, "n_clicks"),
        Input(APPLY_MODEL_MIX_ID, "n_clicks"),
        State({**SLIDER_ID, "channel": ALL}, "id"),
        State({**CONSTRAINT_MIN_ID, "channel": ALL}, "value"),
        State({**CONSTRAINT_MAX_ID, "channel": ALL}, "value"),
        prevent_initial_call=True,
    )
    def _preset_sliders(_reset_clicks, _apply_clicks, ids, min_values, max_values):
        from dash import ctx

        if not ctx.triggered_id:
            raise PreventUpdate

        result = results_by_geo["All"]
        channels = [i["channel"] for i in ids]
        if ctx.triggered_id == APPLY_MODEL_MIX_ID:
            min_weekly = _constraint_dict(channels, min_values)
            max_weekly = _constraint_dict(channels, max_values)
            try:
                rec = recommended_weekly_allocation(
                    result,
                    min_weekly=min_weekly,
                    max_weekly=max_weekly,
                )
            except (RuntimeError, ValueError) as exc:
                status = dmc.Text(str(exc), size="xs", c="red")
                return no_update, no_update, status
            weights = _weights_from_weekly_alloc(result.channels, rec)
            outside = _outside_support_count(result, rec)
            status = dmc.Text(
                "Optimisation re-run with current constraints; "
                f"{outside} channel{'s' if outside != 1 else ''} outside observed support.",
                size="xs",
                c="orange" if outside else "teal",
            )
            fig = _allocation_donut(
                result.channels,
                rec,
                title="Model-suggested",
            )
        else:
            weights = _default_weights(result)
            status = dmc.Text("Reset to current mix.", size="xs", c="dimmed")
            fig = no_update
        return [weights[i["channel"]] for i in ids], fig, status

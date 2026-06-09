from __future__ import annotations

try:
    from .plots import (
        build_calibration_table,
        compute_daily_aggregate_errors,
        compute_daily_procurement_costs,
        compute_rolling_metrics,
        cost_weighted_mae,
        evaluate,
        evaluate_price_buckets,
        hard_switch_predictions,
        spike_weighted_penalty,
        summarize_daily_aggregate_errors,
        underprediction_penalty,
    )
except ImportError:
    from evaluation.plots import (  # type: ignore
        build_calibration_table,
        compute_daily_aggregate_errors,
        compute_daily_procurement_costs,
        compute_rolling_metrics,
        cost_weighted_mae,
        evaluate,
        evaluate_price_buckets,
        hard_switch_predictions,
        spike_weighted_penalty,
        summarize_daily_aggregate_errors,
        underprediction_penalty,
    )

try:
    from ..mcp_pipeline import evaluate_predictions, smape
except ImportError:
    from mcp_pipeline import evaluate_predictions, smape  # type: ignore

__all__ = [
    "build_calibration_table",
    "compute_daily_aggregate_errors",
    "compute_daily_procurement_costs",
    "compute_rolling_metrics",
    "cost_weighted_mae",
    "evaluate",
    "evaluate_predictions",
    "evaluate_price_buckets",
    "hard_switch_predictions",
    "smape",
    "spike_weighted_penalty",
    "summarize_daily_aggregate_errors",
    "underprediction_penalty",
]
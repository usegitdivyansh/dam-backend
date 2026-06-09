from __future__ import annotations

try:
    from ..mcp_pipeline import (
        blend_predictions,
        compute_dynamic_labels,
        compute_regime_diagnostics,
        resolve_regime_label_names,
    )
    from ..evaluation.plots import compute_regime_labels
except ImportError:
    from mcp_pipeline import (  # type: ignore
        blend_predictions,
        compute_dynamic_labels,
        compute_regime_diagnostics,
        resolve_regime_label_names,
    )
    from evaluation.plots import compute_regime_labels  # type: ignore

__all__ = [
    "blend_predictions",
    "compute_dynamic_labels",
    "compute_regime_diagnostics",
    "compute_regime_labels",
    "resolve_regime_label_names",
]
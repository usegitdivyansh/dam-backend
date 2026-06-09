from __future__ import annotations

try:
    from ..mcp_pipeline import build_quantile_model, train_quantile_models
except ImportError:
    from mcp_pipeline import build_quantile_model, train_quantile_models  # type: ignore

__all__ = ["build_quantile_model", "train_quantile_models"]
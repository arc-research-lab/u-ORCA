"""INT8 performance estimates and generator-compatible design search."""

from .perf_model import (
    DATATYPE_CONFIGS,
    estimate_layer,
    estimate_model,
    get_global_aggregation_ns,
    get_stage_params,
)

__all__ = [
    "DATATYPE_CONFIGS", "estimate_layer", "estimate_model",
    "get_global_aggregation_ns", "get_stage_params",
]

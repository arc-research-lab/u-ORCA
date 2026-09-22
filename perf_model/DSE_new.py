"""DeepSets DSE variant that emits u-ORCA-shaped design specifications.

This module deliberately keeps the original performance model's phi search,
padding rules, candidate generation, and cost model.  Its only spatial-search
change is the rho convention: rho is represented physically with M=4 and A=1
instead of padding the logical M=1 to 2*BM.

The generated design is not adjusted to satisfy u-ORCA constraints.  Callers
must validate it and may reject it, but must not silently search a replacement.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Sequence

from DSE import (
    design_space_exploration,
    generate_B_candidates,
    model_param_size_esti,
    pad_dim,
)
from perf_model import (
    DATATYPE_CONFIGS,
    estimate_model,
    get_global_aggregation_ns,
    get_stage_params,
)


GENERATOR_TO_PERF_DATATYPE = {
    "fp32": "fp32_safe",
    "int8": "int8",
    "int4": "int4",
    "int16": "int16",
    "bf16": "bf16",
}


def design_space_exploration_rho(
    in_shapes: Sequence[Sequence[int]],
    params: dict[str, float] | None = None,
    datatype: str = "int8",
    B_limit: int = 38,
) -> tuple[
    list[list[int]] | None,
    float | None,
    float | None,
    float | None,
]:
    """Search rho B partitions while fixing the physical mapping to M=4, A=1."""

    if datatype not in DATATYPE_CONFIGS:
        raise NotImplementedError(f"datatype {datatype} is not supported")
    if B_limit <= 0:
        return None, None, None, None

    datatype_config = DATATYPE_CONFIGS[datatype]
    if params is None:
        params = get_stage_params(datatype, "rho")
    BK = datatype_config["BK"]
    BN = datatype_config["BN"]

    padded_shapes: list[list[int]] = []
    for shape in in_shapes:
        if len(shape) != 3:
            raise ValueError(f"rho shape must contain [M, K, N], got {shape!r}")
        _, K, N = shape
        padded_shapes.append([4, pad_dim(K, BK), pad_dim(N, 2 * BN)])

    B_candidates = [
        [B for B in generate_B_candidates(K, BK) if B <= B_limit]
        for _, K, _ in padded_shapes
    ]
    if not B_candidates or any(not candidates for candidates in B_candidates):
        return None, None, None, None

    best_config: list[list[int]] | None = None
    best_latency = float("inf")
    best_comp: float | None = None
    best_comm: float | None = None
    for B_tuple in itertools.product(*B_candidates):
        config = [
            [M, K, N, 1, B, 1, 1, 1]
            for (M, K, N), B in zip(padded_shapes, B_tuple)
        ]
        latency, comp, comm = estimate_model(config, "cascade", params, datatype)
        if latency < best_latency:
            best_config = config
            best_latency = latency
            best_comp = comp
            best_comm = comm

    if best_config is None:
        return None, None, None, None
    return best_config, best_latency, best_comp, best_comm


def deepsets_estimation_new(
    phi_shapes: Sequence[Sequence[int]],
    rho_shapes: Sequence[Sequence[int]],
    datatype: str = "int8",
    A_limit: int = 8,
    B_limit: int = 37,
) -> dict[str, Any]:
    """Return the lowest-cost original-phi/fixed-M-rho DeepSets design."""

    if datatype not in DATATYPE_CONFIGS:
        raise NotImplementedError(f"datatype {datatype} is not supported")
    if A_limit <= 0 or B_limit <= 1:
        raise ValueError("A_limit must be positive and B_limit must be greater than 1")

    byte_per_element = DATATYPE_CONFIGS[datatype]["byte_per_element"]
    phi_params = get_stage_params(datatype, "phi")
    rho_params = get_stage_params(datatype, "rho")
    best_latency = float("inf")
    best_phi_config: list[list[int]] | None = None
    best_rho_config: list[list[int]] | None = None

    for phi_B_limit in range(1, B_limit):
        rho_B_limit = B_limit - phi_B_limit
        phi_config, _, phi_comp, _ = design_space_exploration(
            phi_shapes, phi_params, datatype, A_limit, phi_B_limit
        )
        rho_config, _, rho_comp, _ = design_space_exploration_rho(
            rho_shapes, rho_params, datatype, rho_B_limit
        )
        if phi_config is None or rho_config is None:
            continue
        if phi_comp is None or rho_comp is None:
            continue
        if sum(layer[4] for layer in phi_config) > phi_B_limit:
            continue
        if sum(layer[4] for layer in rho_config) > rho_B_limit:
            continue

        M, K, _, A, B, C, _, _ = phi_config[0]
        input_elements = math.ceil(M / A) * math.ceil(K / B)
        input_latency = (
            30 + input_elements * byte_per_element / 4 + 4 * (A * C + 2)
        )
        inter_latency = (
            len(phi_config) * phi_params["O_cas"]
            + len(rho_config) * rho_params["O_cas"]
        )
        output_latency = 30 + 64 * byte_per_element / 4
        final_latency = (
            phi_comp + rho_comp + input_latency + inter_latency + output_latency
        ) / 1.25 + get_global_aggregation_ns(datatype)

        if final_latency < best_latency:
            best_latency = final_latency
            best_phi_config = phi_config
            best_rho_config = rho_config

    if best_phi_config is None or best_rho_config is None:
        return {
            "estimated_latency_ns": None,
            "parameter_count": None,
            "phi_config": None,
            "rho_config": None,
        }

    return {
        "estimated_latency_ns": best_latency,
        "parameter_count": model_param_size_esti(best_phi_config + best_rho_config),
        "phi_config": best_phi_config,
        "rho_config": best_rho_config,
    }


def _logical_shapes(model: dict[str, Any], field: str) -> list[list[int]]:
    layers = model.get(field)
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"model {field} must be a non-empty list")
    shapes: list[list[int]] = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"model {field}[{index}] must be an object")
        try:
            shape = [
                int(layer["batch_M"]),
                int(layer["in_features_K"]),
                int(layer["out_features_N"]),
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid logical shape in {field}[{index}]") from exc
        shapes.append(shape)
    return shapes


def build_design_spec(
    model: dict[str, Any],
    *,
    study: str,
    generator_datatype: str,
    precision_labels: Sequence[str],
    source_latency_ns: float,
    methodology: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one complete u-ORCA-shaped spec from one study model record."""

    try:
        perf_datatype = GENERATOR_TO_PERF_DATATYPE[generator_datatype]
    except KeyError as exc:
        supported = ", ".join(GENERATOR_TO_PERF_DATATYPE)
        raise ValueError(
            f"unsupported generator datatype {generator_datatype!r}; expected {supported}"
        ) from exc

    phi_shapes = _logical_shapes(model, "phi_layers")
    rho_shapes = _logical_shapes(model, "rho_layers")
    search_limits = (methodology or {}).get("search_limits", {})
    A_limit = int(search_limits.get("A_limit", 8))
    B_limit = int(search_limits.get("B_limit", 37))
    result = deepsets_estimation_new(
        phi_shapes,
        rho_shapes,
        datatype=perf_datatype,
        A_limit=A_limit,
        B_limit=B_limit,
    )
    if result["phi_config"] is None or result["rho_config"] is None:
        raise ValueError("perf model new found no design within the configured limits")

    variant = model.get("variant")
    if not isinstance(variant, str) or not variant:
        raise ValueError("model variant must be a non-empty string")
    return {
        "datatype": generator_datatype,
        "phi": result["phi_config"],
        "rho": result["rho_config"],
        "metadata": {
            "study": study,
            "variant": variant,
            "precision_labels": list(precision_labels),
            "perf_datatype": perf_datatype,
            "logical_phi_shapes": phi_shapes,
            "logical_rho_shapes": rho_shapes,
            "source_latency_ns": source_latency_ns,
            "dse_new_latency_ns": result["estimated_latency_ns"],
            "dse_new_parameter_count": result["parameter_count"],
            "source_parameter_count": model.get("n_params"),
            "search_limits": {"A_limit": A_limit, "B_limit": B_limit},
        },
    }

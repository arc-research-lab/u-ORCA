"""Search INT8 mappings, skipping candidates rejected by the code generator."""

from __future__ import annotations

import math
from typing import Sequence

from code_gen.spec import (
    INT8_PROFILE, LayerInput, LayerSpec, SpecError, build_mlp_spec, build_model_spec,
    build_stage_specs, require_int8, validate_layer,
)
from .perf_model import (
    AIE_CYCLES_PER_NS, estimate_layer, get_global_aggregation_ns, get_stage_params,
)


def pad_dim(value: int, base: int) -> int:
    """Round up to a power of two that is at least one tile."""
    rounded = math.ceil(value / base) * base
    return 1 << (rounded - 1).bit_length()


def generate_B_candidates(K: int, BK: int) -> list[int]:
    """Keep the original search's power-of-two local K sizes."""
    return [K // width for width in range(BK, K + 1, BK)
            if K % width == 0 and width & (width - 1) == 0]


def _pad_shapes(shapes: Sequence[Sequence[int]], stage: str) -> list[list[int]]:
    padded = []
    for index, shape in enumerate(shapes):
        if len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError(f"{stage} layers must contain three positive integers [M, K, N]")
        M, K, N = shape
        if stage == "rho" and M not in (1, 4):
            raise ValueError("rho logical M must be 1 (or physical M=4)")
        padded.append([
            pad_dim(M, 2 * INT8_PROFILE.BM) if stage != "rho" else 4,
            pad_dim(K, INT8_PROFILE.BK * (2 if stage == "mlp" and index else 1)),
            pad_dim(N, 2 * INT8_PROFILE.BN),
        ])
    # Padding must not hide a disconnected logical network.
    if any(previous[0] != current[0] or previous[2] != current[1]
           for previous, current in zip(shapes, shapes[1:])):
        return []
    return padded


def _boundary_cycles(layer: Sequence[int], stage: str, index: int, count: int) -> float:
    M, K, N, A, B, C, _bias, _relu = layer
    cycles = 0
    if stage in ("phi", "mlp") and index == 0:
        cycles += 30 + (M // A) * (K // B) / 4 + 4 * (A * C + 2)
    if stage == "rho" and index == count - 1:
        cycles += 30 + (M // A) * (N // C) / 4
    if stage == "mlp" and index == count - 1:
        cycles += 30 + (M // A) * (N // C) / 4 + 4 * (A * C + 2)
    return cycles


def _stage_options(shapes, stage, A_limit, B_limit, *, mlp_A=None):
    """Keep the cheapest legal chain for each total B budget.

    Costs are additive, so this dynamic program covers the same combinations
    as full enumeration without keeping every combination in memory.
    """
    padded = _pad_shapes(shapes, stage)
    if len(padded) < (1 if stage == "mlp" else 2):
        return {}
    # Aggregation consumes exactly eight phi rows per partition.
    A = padded[-1][0] // (2 * INT8_PROFILE.BM) if stage == "phi" else 1
    if stage == "mlp":
        A = mlp_A
    if A < 1 or A > A_limit:
        return {}
    base = [[M, K, N, A, 1, 1, 1, 1] for M, K, N in padded]
    try:
        build_stage_specs(base, stage)
    except SpecError:
        return {}

    params = get_stage_params("int8", "phi" if stage == "mlp" else stage)
    states = {0: (0.0, [])}
    for index, (M, K, N) in enumerate(padded):
        candidates = []
        for B in generate_B_candidates(K, INT8_PROFILE.BK):
            if B > B_limit:
                continue
            values = [M, K, N, A, B, 1, 1, 1]
            layer = LayerSpec(
                LayerInput.from_sequence(values),
                INT8_PROFILE.BM, INT8_PROFILE.BK, INT8_PROFILE.BN,
            )
            try:
                validate_layer(layer, stage, index)
            except SpecError:
                continue
            cost = estimate_layer(values, params) + _boundary_cycles(values, stage, index, len(padded))
            candidates.append((B, cost, values))
        next_states = {}
        for used, (cost, config) in states.items():
            for B, layer_cost, values in candidates:
                total = used + B
                if total > B_limit:
                    continue
                candidate_cost = cost + layer_cost
                if total not in next_states or candidate_cost < next_states[total][0]:
                    next_states[total] = (candidate_cost, config + [values])
        states = next_states
    return states


def deepsets_estimation_new(
    phi_shapes, rho_shapes, datatype="int8", A_limit=8, B_limit=37
):
    """Return the cheapest legal design, or None fields when no design fits.

    A_limit limits phi partitions. B_limit limits the sum of B over all phi
    and rho layers. Bias and ReLU are enabled, as in the published DSE.
    """
    require_int8(datatype)
    if any(type(value) is not int or value <= 0 for value in (A_limit, B_limit)):
        raise ValueError("A_limit and B_limit must be positive integers")
    empty = dict(estimated_latency_ns=None, aie_only_latency_ns=None,
                 parameter_count=None, phi_config=None, rho_config=None)
    phi_options = _stage_options(phi_shapes, "phi", A_limit, B_limit)
    rho_options = _stage_options(rho_shapes, "rho", 1, B_limit)
    if not phi_options or not rho_options:
        return empty
    if phi_shapes[-1][2] != rho_shapes[0][1]:
        return empty
    best = None
    for phi_B, (phi_cost, phi) in phi_options.items():
        for rho_B, (rho_cost, rho) in rho_options.items():
            if phi_B + rho_B > B_limit:
                continue
            try:
                build_model_spec(phi, rho)
            except SpecError:
                continue
            cost = phi_cost + rho_cost
            if best is None or cost < best[0]:
                best = (cost, phi, rho)
    if best is None:
        return empty
    cost, phi, rho = best
    compute = sum(estimate_layer(layer, get_stage_params("int8", stage))
                  for stage, layers in (("phi", phi), ("rho", rho)) for layer in layers)
    inter = sum(len(layers) * get_stage_params("int8", stage)["O_cas"]
                for stage, layers in (("phi", phi), ("rho", rho)))
    aggregation = get_global_aggregation_ns()
    return {
        "estimated_latency_ns": (cost + inter) / AIE_CYCLES_PER_NS + aggregation,
        "aie_only_latency_ns": compute / AIE_CYCLES_PER_NS + aggregation,
        "parameter_count": sum(layer[1] * layer[2] + layer[2] * layer[6] for layer in phi + rho),
        "phi_config": phi,
        "rho_config": rho,
    }


def mlp_estimation_new(layers, datatype="int8", A_limit=8, B_limit=37):
    """Search the historical standalone MLP chain over a common A and total B."""
    require_int8(datatype)
    if any(type(value) is not int or value <= 0 for value in (A_limit, B_limit)):
        raise ValueError("A_limit and B_limit must be positive integers")
    padded = _pad_shapes(layers, "mlp")
    empty = dict(estimated_latency_ns=None, aie_only_latency_ns=None,
                 parameter_count=None, layers_config=None)
    if not padded:
        return empty
    best = None
    for A in range(1, min(A_limit, padded[0][0] // (2 * INT8_PROFILE.BM)) + 1):
        if padded[0][0] % (A * 2 * INT8_PROFILE.BM):
            continue
        for cost, config in _stage_options(layers, "mlp", A_limit, B_limit, mlp_A=A).values():
            try:
                build_mlp_spec(config)
            except SpecError:
                continue
            if best is None or cost < best[0]:
                best = cost, config
    if best is None:
        return empty
    cost, config = best
    params = get_stage_params("int8", "phi")
    compute = sum(estimate_layer(layer, params) for layer in config)
    return {
        "estimated_latency_ns": (cost + (len(config) - 1) * params["O_cas"]) / AIE_CYCLES_PER_NS,
        "aie_only_latency_ns": compute / AIE_CYCLES_PER_NS,
        "parameter_count": sum(layer[1] * layer[2] + layer[2] * layer[6] for layer in config),
        "layers_config": config,
    }


def build_design_spec(model, *, A_limit=8, B_limit=37):
    """Convert a logical model JSON object into a code-generation spec."""
    if not isinstance(model, dict):
        raise SpecError("model JSON must be an object")
    datatype = model.get("datatype", "int8")
    model_type = model.get("model_type", "mlp" if "layers" in model else "deepsets")
    if model_type == "mlp":
        if "phi" in model or "rho" in model:
            raise SpecError("MLP input uses layers, without phi or rho")
        result = mlp_estimation_new(model["layers"], datatype, A_limit, B_limit)
        if result["layers_config"] is None:
            raise SpecError("No generator-compatible INT8 MLP design fits these shapes and search limits")
        spec = {
            "graph_name": model.get("graph_name", "UOrcaGraph"),
            "model_type": "mlp",
            "datatype": "int8",
            "layers": result["layers_config"],
            "metadata": {
                "logical_layers": model["layers"],
                "A_limit": A_limit, "B_limit": B_limit,
                **{key: value for key, value in result.items() if key != "layers_config"},
            },
        }
        build_mlp_spec(spec["layers"], graph_name=spec["graph_name"])
        return spec
    if model_type != "deepsets" or "layers" in model:
        raise SpecError("model_type must be mlp (layers) or deepsets (phi and rho)")
    result = deepsets_estimation_new(model["phi"], model["rho"], datatype, A_limit, B_limit)
    if result["phi_config"] is None:
        raise SpecError("No generator-compatible INT8 design fits these shapes and search limits")
    spec = {
        "graph_name": model.get("graph_name", "UOrcaGraph"),
        "model_type": "deepsets",
        "datatype": "int8",
        "phi": result["phi_config"],
        "rho": result["rho_config"],
        "metadata": {
            "logical_phi": model["phi"], "logical_rho": model["rho"],
            "A_limit": A_limit, "B_limit": B_limit,
            **{key: value for key, value in result.items() if key not in ("phi_config", "rho_config")},
        },
    }
    build_model_spec(spec["phi"], spec["rho"], graph_name=spec["graph_name"])
    return spec

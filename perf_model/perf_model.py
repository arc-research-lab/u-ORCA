"""The calibrated cost model. Layer estimates use AIE cycles."""

import math

from code_gen.spec import DATATYPE_PROFILES, resolve_datatype_profile

DATATYPE_CONFIGS = {
    name: {
        "BM": profile.BM, "BK": profile.BK, "BN": profile.BN,
        # FP32 uses the historical trace-calibrated fp32_safe cost profile.
        "mac_cycles": 38 if name == "fp32" else 1,
        "byte_per_element": profile.activation_bytes,
    } for name, profile in DATATYPE_PROFILES.items()
}
CALIBRATED_STAGE_PARAMS = {
    "int4": {
        "phi": {"L_setup": 30, "L_o": 1, "O_cas": 0},
        "rho": {"L_setup": 1, "L_o": 19, "O_cas": 0},
    },
    "int8": {
        "phi": {"L_setup": 15, "L_o": 9, "O_cas": 0},
        "rho": {"L_setup": 18, "L_o": 10, "O_cas": 0},
    },
    "int16": {
        "phi": {"L_setup": 18, "L_o": 13, "O_cas": 0},
        "rho": {"L_setup": 42, "L_o": 3, "O_cas": 0},
    },
    "bf16": {
        "phi": {"L_setup": 28, "L_o": 1, "O_cas": 0},
        "rho": {"L_setup": 2, "L_o": 44, "O_cas": 0},
    },
    "fp32": {
        "phi": {"L_setup": 12, "L_o": 1, "O_cas": 0},
        "rho": {"L_setup": 0, "L_o": 237, "O_cas": 0},
    },
}
AIE_CYCLES_PER_NS = 1.25
GLOBAL_AGGREGATION_NS = {datatype: 150 for datatype in DATATYPE_PROFILES}


def get_stage_params(datatype, stage):
    """Return calibrated parameters for phi or rho."""
    resolve_datatype_profile(datatype)
    if stage not in ("phi", "rho"):
        raise ValueError("stage must be phi or rho")
    return dict(CALIBRATED_STAGE_PARAMS[datatype][stage])


def get_global_aggregation_ns(datatype="int8"):
    resolve_datatype_profile(datatype)
    return GLOBAL_AGGREGATION_NS[datatype]


def estimate_layer(in_param, params, datatype="int8"):
    """Estimate cycles for [M, K, N, A, B, C, bias, relu]."""
    resolve_datatype_profile(datatype)
    M, K, N, A, B, C, bias, _relu = in_param
    profile = DATATYPE_CONFIGS[datatype]
    h1, w1, w2 = math.ceil(M / A), math.ceil(K / B), math.ceil(N / C)
    if "L_setup" in params:
        setup = params["L_setup"]
    else:
        setup = params["L_epi"] + params["L_cas"] + (params["L_br"] if bias else 0)
    inner = 4 * w1 / profile["BK"] * profile["mac_cycles"] + setup
    return (h1 * w2 / (4 * profile["BM"] * profile["BN"]) + B - 1) * inner + params["L_o"]


def estimate_model(in_params, mode, configs, datatype="int8"):
    """Return (total, compute, communication) in AIE cycles for one stage."""
    profile = resolve_datatype_profile(datatype)
    if mode not in ("pl", "shared_mem", "direct", "cascade"):
        raise ValueError("mode must be pl, shared_mem, direct, or cascade")
    if not in_params:
        raise ValueError("at least one layer is required")
    compute = sum(estimate_layer(layer, configs, datatype) for layer in in_params)
    inputs, outputs = [], []
    for M, K, N, A, B, C, _bias, _relu in in_params:
        distance = A * C + 2
        if mode == "shared_mem":
            distance = A * C
        elif mode == "direct":
            distance = B
        rows = math.ceil(M / A)
        inputs.append(30 + rows * math.ceil(K / B) * profile.activation_bytes / 4 + 4 * distance)
        outputs.append(30 + rows * math.ceil(N / C) * profile.activation_bytes / 4 + 4 * distance)
    if mode in ("pl", "shared_mem"):
        communication = sum(inputs) + sum(outputs)
        overhead = len(in_params) * (2 * 1250 / 300 if mode == "pl" else 1)
    else:
        communication = inputs[0] + outputs[-1]
        overhead = 0
        if mode == "direct":
            communication += sum(max(outputs[i], inputs[i + 1]) for i in range(len(in_params) - 1))
        else:
            communication += (len(in_params) - 1) * configs["O_cas"]
    return compute + communication + overhead, compute, communication

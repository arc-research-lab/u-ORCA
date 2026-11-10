"""The published, calibrated INT8 cost model. Layer estimates use AIE cycles."""

import math

from code_gen.spec import INT8_PROFILE, require_int8

DATATYPE_CONFIGS = {
    "int8": {
        "BM": INT8_PROFILE.BM, "BK": INT8_PROFILE.BK, "BN": INT8_PROFILE.BN,
        "mac_cycles": 1, "byte_per_element": 1,
    }
}
CALIBRATED_STAGE_PARAMS = {
    "int8": {
        "phi": {"L_setup": 15, "L_o": 9, "O_cas": 0},
        "rho": {"L_setup": 18, "L_o": 10, "O_cas": 0},
    }
}
AIE_CYCLES_PER_NS = 1.25
GLOBAL_AGGREGATION_NS = {"int8": 150}


def get_stage_params(datatype, stage):
    """Return the published INT8 parameters for phi or rho."""
    require_int8(datatype)
    if stage not in ("phi", "rho"):
        raise ValueError("stage must be phi or rho")
    return dict(CALIBRATED_STAGE_PARAMS[datatype][stage])


def get_global_aggregation_ns(datatype="int8"):
    require_int8(datatype)
    return GLOBAL_AGGREGATION_NS[datatype]


def estimate_layer(in_param, params, datatype="int8"):
    """Estimate cycles for [M, K, N, A, B, C, bias, relu]."""
    require_int8(datatype)
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
    require_int8(datatype)
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
        inputs.append(30 + rows * math.ceil(K / B) / 4 + 4 * distance)
        outputs.append(30 + rows * math.ceil(N / C) / 4 + 4 * distance)
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

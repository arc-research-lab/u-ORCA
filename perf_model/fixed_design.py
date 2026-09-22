"""Estimate every datatype for one fixed DeepSets spatial design."""

import math

from DSE import pad_dim
from perf_model import (
    DATATYPE_CONFIGS,
    estimate_model,
    get_global_aggregation_ns,
    get_stage_params,
)


PHI_SHAPES = [
    [64, 21, 64],
    [64, 64, 64],
    [64, 64, 64],
]
RHO_SHAPES = [
    [1, 64, 64],
    [1, 64, 10],
]

PHI_A = 8
PHI_B = [2, 2, 2]
RHO_A = 1
RHO_B = [4, 1]

AIE_CYCLES_PER_NS = 1.25
DMA_OVERHEAD = 30
DMA_BYTES_PER_CYCLE = 4
FINAL_OUTPUT_ELEMENTS = 64  # 1x10 is padded to 4x16 by the model.


def make_config(shapes, a, b_values, datatype):
    datatype_config = DATATYPE_CONFIGS[datatype]
    bm = datatype_config["BM"]
    bk = datatype_config["BK"]
    bn = datatype_config["BN"]

    config = []
    for (m, k, n), b in zip(shapes, b_values):
        padded_m = pad_dim(m, 2 * bm)
        padded_k = pad_dim(k, bk)
        padded_n = pad_dim(n, 2 * bn)
        config.append([padded_m, padded_k, padded_n, a, b, 1, 1, 1])
    return config


def estimate_fixed_design(datatype):
    bytes_per_element = DATATYPE_CONFIGS[datatype]["byte_per_element"]
    phi_params = get_stage_params(datatype, "phi")
    rho_params = get_stage_params(datatype, "rho")
    phi_config = make_config(PHI_SHAPES, PHI_A, PHI_B, datatype)
    rho_config = make_config(RHO_SHAPES, RHO_A, RHO_B, datatype)

    _, phi_comp_cycles, _ = estimate_model(
        phi_config, "cascade", phi_params, datatype
    )
    _, rho_comp_cycles, _ = estimate_model(
        rho_config, "cascade", rho_params, datatype
    )

    m, k, _, a, b, c, _, _ = phi_config[0]
    input_elements = math.ceil(m / a) * math.ceil(k / b)
    input_cycles = (
        DMA_OVERHEAD
        + input_elements * bytes_per_element / DMA_BYTES_PER_CYCLE
        + 4 * (a * c + 2)
    )

    inter_cycles = (
        len(phi_config) * phi_params["O_cas"]
        + len(rho_config) * rho_params["O_cas"]
    )
    output_cycles = (
        DMA_OVERHEAD
        + FINAL_OUTPUT_ELEMENTS * bytes_per_element / DMA_BYTES_PER_CYCLE
    )

    compute_ns = (phi_comp_cycles + rho_comp_cycles) / AIE_CYCLES_PER_NS
    communication_ns = (input_cycles + inter_cycles + output_cycles) / AIE_CYCLES_PER_NS
    global_aggregation_ns = get_global_aggregation_ns(datatype)
    total_ns = compute_ns + communication_ns + global_aggregation_ns

    return {
        "datatype": datatype,
        "phi_config": phi_config,
        "rho_config": rho_config,
        "compute_ns": compute_ns,
        "communication_ns": communication_ns,
        "ga_ns": global_aggregation_ns,
        "total_ns": total_ns,
    }


def main():
    print("Fixed design: Phi A=8, B=[2,2,2]; Rho A=1, B=[4,1]")
    print("datatype       compute(ns)    comm(ns)      GA(ns)   total(ns)")
    print("-" * 68)
    for datatype in DATATYPE_CONFIGS:
        result = estimate_fixed_design(datatype)
        print(
            f"{datatype:<14}"
            f"{result['compute_ns']:>11.1f}"
            f"{result['communication_ns']:>12.1f}"
            f"{result['ga_ns']:>12.1f}"
            f"{result['total_ns']:>12.1f}"
        )


if __name__ == "__main__":
    main()

import math

DATATYPE_CONFIGS = {
    "int4": {"BM": 4, "BK": 16, "BN": 8, "mac_cycles": 1, "byte_per_element": 1},
    "int8": {"BM": 4, "BK": 8, "BN": 8, "mac_cycles": 1, "byte_per_element": 1},
    "int16": {"BM": 4, "BK": 4, "BN": 4, "mac_cycles": 1, "byte_per_element": 2},
    "int32": {"BM": 4, "BK": 2, "BN": 4, "mac_cycles": 2, "byte_per_element": 4},
    "bf16": {"BM": 4, "BK": 8, "BN": 4, "mac_cycles": 1, "byte_per_element": 2},
    "fp32_fast": {"BM": 4, "BK": 8, "BN": 4, "mac_cycles": 3, "byte_per_element": 4},
    "fp32_mid": {"BM": 4, "BK": 8, "BN": 4, "mac_cycles": 6, "byte_per_element": 4},
    # Pre-calibration backup:
    # "fp32_safe": {"BM": 4, "BK": 8, "BN": 4, "mac_cycles": 9, "byte_per_element": 4},
    "fp32_safe": {"BM": 4, "BK": 8, "BN": 4, "mac_cycles": 38, "byte_per_element": 4},
}

# Pre-calibration, datatype-independent stage-parameter backup:
# params_phi = {
#     "L_epi": 2,
#     "L_br": 1,
#     "L_o": 0,
#     "L_cas": 14,
#     "O_cas": 0,
# }
# params_rho = {
#     "L_epi": 2,
#     "L_br": 2,
#     "L_o": 59,
#     "L_cas": 8,
#     "O_cas": 0,
# }
#
# Keep the original public dictionaries for callers that explicitly pass a
# legacy profile and as the fallback for datatypes without a calibrated
# profile.  The calibrated DSE path uses get_stage_params() below.
params_phi = {
    "L_epi": 2,
    "L_br":1,
    "L_o": 0,
    "L_cas": 14,
    "O_cas": 0
}
params_rho = {
    "L_epi": 2,
    "L_br":2,
    "L_o": 59,
    "L_cas": 8,
    "O_cas": 0
}


# All available calibration designs enable bias, so L_epi, L_br, and L_cas
# are not separately identifiable: only their sum contributes to a trace.
# Calibrated profiles therefore expose that identifiable sum as L_setup.
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
    "fp32_safe": {
        "phi": {"L_setup": 12, "L_o": 1, "O_cas": 0},
        "rho": {"L_setup": 0, "L_o": 237, "O_cas": 0},
    },
}

GLOBAL_AGGREGATION_NS = {
    datatype: 150 for datatype in CALIBRATED_STAGE_PARAMS
}


def get_stage_params(datatype, stage):
    """Return calibrated phi/rho parameters for one performance datatype."""
    if stage not in ("phi", "rho"):
        raise ValueError(f"unknown stage {stage!r}; expected 'phi' or 'rho'")
    profile = CALIBRATED_STAGE_PARAMS.get(datatype)
    if profile is None:
        return dict(params_phi if stage == "phi" else params_rho)
    return dict(profile[stage])


def get_global_aggregation_ns(datatype):
    """Return the datatype profile's fixed global-aggregation latency."""
    return GLOBAL_AGGREGATION_NS.get(datatype, 150)

def estimate_layer(in_param, p, datatype="int8"):
    """estimat the computation time of a layer
    - input: 
        - in_param=[M, K, N, A, B, C, bias, relu], 
            - M,K,N: MM shape, 
            - A,B,C: AIE array shape
            - bias and relu: 1 for enable
        -p: hyper parameters of each computation stage
        -datatype: datatype of the kernel
    - output: computation latency in cycles"""
    if datatype not in DATATYPE_CONFIGS:
        raise NotImplementedError(f"datatype {datatype} is not supported")
    datatype_config = DATATYPE_CONFIGS[datatype]
    # parse input
    M, K, N, A, B, C, bias, relu = in_param
    
    BM = datatype_config["BM"]
    BK = datatype_config["BK"]
    BN = datatype_config["BN"]
    mac_cycles = datatype_config["mac_cycles"]
    L_o = p["L_o"]

    # select hyperparam
    h1 = math.ceil(M / A)
    w1 = math.ceil(K / B)
    w2 = math.ceil(N / C)

    if "L_setup" in p:
        L_setup = p["L_setup"]
    else:
        L_setup = p["L_epi"] + p["L_cas"]
        if bias == 1:
            L_setup += p["L_br"]

    if B>1:
        Lj = 4 * w1 / BK * mac_cycles + L_setup
    else:
        Lj = 4 * w1 / BK * mac_cycles + L_setup
    L_comp = (h1 * w2 / 4 / BM / BN + B - 1) * Lj + L_o

    return L_comp

def estimate_model(in_params,mode,configs,datatype="int8"):
    """estimate the performance of a model
    - in_params: [[L1 spec],[L2 spec],...]
    - mode: one in ["pl","shared_mem","direct","cascade"]
    - configs: hyper parameters
    - output: total latency, total communication latency, total output latency"""
    AIE_freq = 1250
    PL_Freq = 300
    BR_lat = 2
    Total_PLIO = 224
    DMA_BW = 4 #byte/cycle
    DMA_ovhd = 30#30 cycles to init DMA
    Cas_ovhd = configs["O_cas"]#4 cycles for cascade stall
    assert mode in ["pl","shared_mem","direct","cascade"]
    if datatype not in DATATYPE_CONFIGS:
        raise NotImplementedError(f"datatype {datatype} is not supported")
    byte_per_element = DATATYPE_CONFIGS[datatype]["byte_per_element"]
    comp_lats = [estimate_layer(l,configs,datatype) for l in in_params]
    in_lats = []
    out_lats = []
    #comp latency for each layer
    for layer in in_params:
        M,K,N,A,B,C,bias,relu = layer
        h1 = math.ceil(M/A)
        w1 = math.ceil(K/B)
        w2 = math.ceil(N/C)
        comm_in = h1*w1*byte_per_element#Byte
        comm_out = h1*w2*byte_per_element
        if mode == 'pl': max_comm_distance = A*C+2
        elif mode == 'shared_mem': max_comm_distance = A*C
        elif mode == 'direct':max_comm_distance = B
        elif mode == 'cascade':max_comm_distance = A*C+2
        #input and output latency
        in_lat = DMA_ovhd + comm_in/DMA_BW + 4*max_comm_distance
        out_lat = DMA_ovhd + comm_out/DMA_BW + 4*max_comm_distance
        in_lats.append(in_lat)
        out_lats.append(out_lat)
    #comput overall latency
    comp = sum(comp_lats)
    if mode == 'pl':#add bias relu latency
        lat = sum(comp_lats)+sum(in_lats)+sum(out_lats)+len(in_params)*BR_lat*AIE_freq/PL_Freq
        comm = sum(in_lats)+sum(out_lats)
    if mode == 'shared_mem':
        lat = sum(comp_lats)+sum(in_lats)+sum(out_lats)+len(in_params)
        comm = sum(in_lats)+sum(out_lats)
    if mode == 'direct': #only one DMA in between, bounded by the larger one
        lat = sum(comp_lats)
        comm = 0
        lat += in_lats[0] + out_lats[-1]
        comm += in_lats[0] + out_lats[-1]
        for i in range(len(in_params)-1):
            lat += max(out_lats[i],in_lats[i+1])
            comm += max(out_lats[i],in_lats[i+1])
    if mode == 'cascade':#using cascade for comm
        lat = sum(comp_lats)
        lat += in_lats[0] + out_lats[-1]
        comm = in_lats[0] + out_lats[-1]
        for i in range(len(in_params)-1):
            lat += Cas_ovhd
            comm +=Cas_ovhd
    return lat, comp, comm

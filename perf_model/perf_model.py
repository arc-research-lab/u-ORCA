import math

params_phi = {
    "BM":4,
    "BK":8,
    "BN":8,
    "L_epi": 2,
    "L_br":1,
    "L_o": 0,
    "L_cas": 14,
    "O_cas": 0
}
params_rho = {
    "BM":4,
    "BK":8,
    "BN":8,
    "L_epi": 2,
    "L_br":2,
    "L_o": 59,
    "L_cas": 8,
    "O_cas": 0
}

def estimate_layer(in_param, p):
    """estimat the computation time of a layer
    - input: 
        - in_param=[M, K, N, A, B, C, bias, relu], 
            - M,K,N: MM shape, 
            - A,B,C: AIE array shape
            - bias and relu: 1 for enable
        -p: hyper parameters of each computation stage
    - output: computation latency in cycles"""
    # parse input
    M, K, N, A, B, C, bias, relu = in_param
    
    BM = p["BM"]
    BK = p["BK"]
    BN = p["BN"]
    L_epi = p["L_epi"]
    L_br = p["L_br"]
    L_cas = p["L_cas"]
    L_o = p["L_o"]

    # select hyperparam
    h1 = math.ceil(M / A)
    w1 = math.ceil(K / B)
    w2 = math.ceil(N / C)

    L_epi_eff = L_epi
    if bias == 1:
        L_epi_eff += L_br

    if B>1:
        Lj = 4 * w1 / BK + L_epi_eff + L_cas 
    else:
        Lj = 4 * w1 / BK + L_epi_eff + L_cas
    L_comp = (h1 * w2 / 4 / BM / BN + B - 1) * Lj + L_o

    return L_comp

def estimate_model(in_params,mode,configs):
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
    comp_lats = [estimate_layer(l,configs) for l in in_params]
    in_lats = []
    out_lats = []
    #comp latency for each layer
    for layer in in_params:
        M,K,N,A,B,C,bias,relu = layer
        h1 = math.ceil(M/A)
        w1 = math.ceil(K/B)
        w2 = math.ceil(N/C)
        comm_in = h1*w1#Byte
        comm_out = h1*w2
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

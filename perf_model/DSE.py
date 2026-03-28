import itertools
import math
from perf_model import estimate_model, params_phi, params_rho

def next_power_of_two(x):
    """return the next power of two >= x"""
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))
def pad_dim(x, base):
    """
    pad x to be:
    (1) multiple of base
    (2) power of two
    """
    x = math.ceil(x / base) * base
    return next_power_of_two(x)
def is_power_of_two(x):
    return (x & (x - 1)) == 0
def generate_A_candidates(M, BM):
    """
    generate Ai such that:
    M / Ai = tile_M
    tile_M is power of 2 and multiple of 2*BM
    """
    candidates = []

    for tile_M in range(2 * BM, M + 1, 2 * BM):
        if is_power_of_two(tile_M) and M % tile_M == 0:
            Ai = M // tile_M
            candidates.append(Ai)

    return candidates
def generate_B_candidates(K, BK):
    """
    generate Bi such that:
    K / Bi = tile_K
    tile_K is power of 2 and multiple of BK
    """
    candidates = []

    for tile_K in range(BK, K + 1, BK):
        if is_power_of_two(tile_K) and K % tile_K == 0:
            Bi = K // tile_K
            candidates.append(Bi)

    return candidates

def design_space_exploration(in_shapes, params):
    """
    in_shapes: [[M1,K1,N1],[M2,K2,N2],...]
    return best_config, best_latency
    """

    BM = params["BM"]
    BK = params["BK"]
    BN = params["BN"]

    # -----------------------------
    # Step 1: padding
    # -----------------------------
    padded_shapes = []

    for M, K, N in in_shapes:
        Mp = pad_dim(M, 2 * BM)
        Kp = pad_dim(K, BK)
        Np = pad_dim(N, 2 * BN)

        padded_shapes.append([Mp, Kp, Np])

    # -----------------------------
    # Step 2: Ai candidates
    # Ai must work for all layers
    # -----------------------------
    A_sets = []

    for M, K, N in padded_shapes:
        A_sets.append(set(generate_A_candidates(M, BM)))

    A_candidates = sorted(set.intersection(*A_sets))

    best_lat = float("inf")
    best_config = None

    # -----------------------------
    # Step 3: enumerate Ai
    # -----------------------------
    for A in A_candidates:
        # B candidates per layer
        B_candidates = []
        for M, K, N in padded_shapes:
            B_candidates.append(generate_B_candidates(K, BK))
        # print(B_candidates)
        for B_tuple in itertools.product(*B_candidates):
            config = []
            for i, (M, K, N) in enumerate(padded_shapes):
                B = B_tuple[i]
                config.append([M, K, N, A, B, 1, 1, 1])

            lat, comp, comm = estimate_model(config, 'cascade', params)

            if lat < best_lat:
                best_lat = lat
                best_comp = comp
                best_comm = comm
                best_config = config

    return best_config, best_lat, best_comp, best_comm

def model_param_size_esti(shapes):
    total_param_num = 0
    for layer in shapes:
        M,K,N,A,B,C,bias,relu = layer
        total_param_num += K*N #linear
        if bias == 1:
            total_param_num += N #bias
    return total_param_num




def deepsets_estimation(phi_shapes,rho_shapes):
    DMA_ovhd = 30#30 cycles to init DMA
    DMA_BW = 4 #byte/cycle
    #search the best config for phi and rho layers
    phi_config,phi_lat,phi_comp,phi_comm= design_space_exploration(phi_shapes,params_phi)
    rho_config,rho_lat,rho_comp,rho_comm= design_space_exploration(rho_shapes,params_rho)
    #as the AIEs shouldn't over utilize, only check the B dim
    total_B=1#GA layer
    for layer in phi_config+rho_config:
        total_B+=layer[4]
    assert total_B <= 38, f"too many AIEs utilized in the col dim! expect <=38, get{total_B}"
    #compute final latency
    final_lat = phi_comp + rho_comp 
    #add input comm lat
    layer = phi_config[0]
    M,K,N,A,B,C,bias,relu = layer
    h1 = math.ceil(M/A)
    w1 = math.ceil(K/B)
    comm_in = h1*w1#Byte
    max_comm_distance = A*C+2
    in_lat = DMA_ovhd + comm_in/DMA_BW + 4*max_comm_distance
    #add inter layer comm lat
    inter_lat = len(phi_config)*params_phi["O_cas"]+len(rho_config)*params_rho["O_cas"]
    #add out lat
    comm_out=64#pad 1x10 --> 4x16
    out_lat = DMA_ovhd + comm_out/DMA_BW
    final_lat += in_lat + inter_lat + out_lat
    #convert from cycles to ns
    final_lat /= 1.25
    #add GA layers: for an MxN mat the latency almost do not change with N dim
    final_lat+=150
    final_param = model_param_size_esti(phi_config+rho_config)
    return final_lat, final_param
if __name__ == "__main__":
    model = [[64,64,64] for i in range(3)]
    best_config, best_lat = design_space_exploration(model,params_phi)
    print('best_config',best_config)
    print('best_latency',best_lat)
    print(estimate_model([[64,64,64,8,2,1,1,1] for i in range(3)],'cascade',params_phi))
import itertools
import csv
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from DSE import deepsets_estimation

# ========================
# config
# ========================
dims = [32, 64, 128]  
L1_min = 4                  
L1_max = 6                  
L2_min = 2                  
L2_max = 4                  
cpu_num = 48

# ========================
# phi layer
# ========================
def generate_phi_configs():
    configs = []
    for n_layers in range(L1_min, L1_max+1):
        for hidden in itertools.product(dims, repeat=n_layers):
            layers = []
            in_dim = 21
            for h in hidden:
                layers.append([64, in_dim, h])
                in_dim = h
            configs.append(layers)
    return configs

# ========================
# 生成 rho 层组合
# ========================
def generate_rho_configs(K):
    configs = []
    for n_layers in range(L2_min, L2_max+1):
        if n_layers == 1:
            configs.append([[1, K, 10]])
            continue
        for hidden in itertools.product(dims, repeat=n_layers-1):
            layers = []
            in_dim = K
            for h in hidden:
                layers.append([1, in_dim, h])
                in_dim = h
            layers.append([1, in_dim, 10])
            configs.append(layers)
    return configs

# ========================
# 单个 phi 配置评估
# ========================
def evaluate_phi(phi):
    K = phi[-1][-1]
    rho_configs = generate_rho_configs(K)
    results = []
    for rho in rho_configs:
        try:
            lat, param = deepsets_estimation(phi, rho)
            results.append({
                "phi": str(phi),
                "rho": str(rho),
                "latency": lat,
                "param": param
            })
        except Exception:
            continue
    return results

# ========================
# 主搜索逻辑
# ========================
if __name__ == "__main__":
    phi_configs = generate_phi_configs()
    print(f"Total phi configs (with L1_min={L1_min}): {len(phi_configs)}")

    pool = Pool(processes=min(cpu_num, cpu_count()))
    all_results_nested = pool.map(evaluate_phi, phi_configs)
    pool.close()
    pool.join()

    # 展平成列表
    all_results = [item for sublist in all_results_nested for item in sublist]
    print(f"Total design points evaluated: {len(all_results)}")

    # ========================
    # 保存 CSV
    # ========================
    csv_file = "search_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phi", "rho", "latency", "param"])
        writer.writeheader()
        writer.writerows(all_results)
    print("Saved to", csv_file)

    # ========================
    # 绘图
    # ========================
    latencies = [r["latency"] for r in all_results]
    params = [r["param"] for r in all_results]

    plt.figure()
    plt.scatter(params, latencies, s=10)
    plt.xlabel("Parameter Size")
    plt.ylabel("Latency (ns)")
    plt.title("Latency vs Parameter Size")
    plt.grid(True)
    plt.savefig("figure.png", dpi=300)
    print("Figure saved as figure.png")
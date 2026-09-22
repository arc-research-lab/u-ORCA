"""
main func for deepsets model 
"""
from DSE import deepsets_estimation
from perf_model import DATATYPE_CONFIGS
#######################
#INPUTS
#######################
# phi_layers = [
#     [64,21,64],
#     [64,64,64],
#     [64,64,64]
# ]
# rho_layers = [
#     [1,64,64],
#     [1,64,10]
# ]

phi_layers = [
    # [64,21,32],
    # [64,32,32],
    # [64,32,32]
    [64,21,64],
    [64,64,64],
    [64,64,64]
]
rho_layers = [
    [1,64,64],
    # [1,64,64],
    # [1,64,64],
    [1,64,10]
]


for datatype in DATATYPE_CONFIGS:
    lat, param = deepsets_estimation(phi_layers, rho_layers, datatype=datatype)
    print(f'datatype: {datatype}')
    print(f'estimated latency: {lat} ns')
    print(f'param size: {param}')
    print('----------------------')

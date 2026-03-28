"""
main func for deepsets model 
"""
from DSE import deepsets_estimation
#######################
#INPUTS
#######################
phi_layers = [
    [64,21,64],
    [64,64,64],
    [64,64,64]
]
rho_layers = [
    [1,64,64],
    [1,64,10]
]

lat, param = deepsets_estimation(phi_layers,rho_layers)
print(f'estimated latency: {lat} ns')
print(f'param size: {param}')
from perf_model import estimate_model, params_phi, params_rho
import math
# print(estimate_model([[64,32,32,8,3,1,1,1] for i in range(3)], 'cascade', params_phi))
lat_phi, comp_phi, comm_phi = estimate_model([[64,32,32,8,2,1,1,1] for i in range(3)], 'cascade', params_phi)
print(lat_phi, comp_phi, comm_phi)
lat_rho, comp_rho, comm_rho = estimate_model([[8,32,32,1,2,1,1,1] for i in range(2)]+[[8,32,16,1,1,1,1,1]], 'cascade', params_phi)
print(lat_rho, comp_rho, comm_rho)
final_lat = comp_phi + comp_rho 
layer=[64,32,32,8,2,1,1,1]
M,K,N,A,B,C,bias,relu = layer
h1 = math.ceil(M/A)
w1 = math.ceil(K/B)
comm_in = h1*w1#Byte
max_comm_distance = A*C+2
DMA_ovhd = 30#30 cycles to init DMA
DMA_BW = 4 #byte/cycle
in_lat = DMA_ovhd + comm_in/DMA_BW + 4*max_comm_distance
inter_lat = 3*params_phi["O_cas"]+3*params_rho["O_cas"]
comm_out=64#pad 1x10 --> 4x16
out_lat = DMA_ovhd + comm_out/DMA_BW
final_lat += in_lat + inter_lat + out_lat
final_lat /= 1.25
final_lat+=150
print(final_lat)

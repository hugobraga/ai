# dsa_script-05
# Obter os detalhes da GPU

# Import
from numba import cuda

# Obtém o dispositivo CUDA atual
device = cuda.get_current_device()

# Consulta várias propriedades do dispositivo CUDA
max_threads_per_block = device.MAX_THREADS_PER_BLOCK  # Máximo de threads por bloco
max_block_dim_x = device.MAX_BLOCK_DIM_X  # Máxima dimensão X de um bloco
max_block_dim_y = device.MAX_BLOCK_DIM_Y  # Máxima dimensão Y de um bloco
max_block_dim_z = device.MAX_BLOCK_DIM_Z  # Máxima dimensão Z de um bloco
max_grid_dim_x = device.MAX_GRID_DIM_X  # Máxima dimensão X de uma grid
max_grid_dim_y = device.MAX_GRID_DIM_Y  # Máxima dimensão Y de uma grid
max_grid_dim_z = device.MAX_GRID_DIM_Z  # Máxima dimensão Z de uma grid
name = device.name  # Nome do dispositivo
compute_capability = device.compute_capability  # Capacidade de computação do dispositivo
warp_size = device.WARP_SIZE  # Tamanho do warp, uma unidade de execução em CUDA
multiprocessors = device.MULTIPROCESSOR_COUNT  # Número de multiprocessadores

# Imprime as propriedades
print("Modelo da GPU:", name)
print("Compute Capability:", compute_capability)
print("Max threads per block:", max_threads_per_block)
print("Warp Size:", warp_size)
print("Number of Multiprocessors:", multiprocessors)
print("Max block dimensions (x, y, z):", max_block_dim_x, max_block_dim_y, max_block_dim_z)
print("Max grid dimensions (x, y, z):", max_grid_dim_x, max_grid_dim_y, max_grid_dim_z)

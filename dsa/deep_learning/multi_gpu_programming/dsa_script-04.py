# dsa_script-04.py
# Execução em GPU e CPU

# Importando o módulo numpy como np para operações matemáticas e manipulação de arrays
import numpy as np

# Importando a função default_timer do módulo timeit e renomeando-a como timer
from timeit import default_timer as timer

# Importando funções e decoradores do módulo numba para compilação just-in-time (JIT) e suporte a CUDA
from numba import cuda, jit, float64, void, prange

# Importando a função perf_counter_ns do módulo time
from time import perf_counter_ns

# Função para incrementar cada elemento de um array, será executada na CPU sem otimização
def FillArrayWithCPU(a):
    for k in range(10000000):
        a[k] += 1

# Função para incrementar elementos de um array, otimizada com JIT e paralelização, executada em múltiplos Cores na CPU
@jit(void(float64[:]), nopython=True, parallel=True)
def FillArrayWithJITParallel(a):
    for k in prange(10000000):
        a[k] += 1

# Função para incrementar elementos de um array, otimizada com JIT e destinada a executar em uma GPU usando CUDA
@jit(void(float64[:]), nopython=True, target_backend='cuda')
def FillArrayWithGPU(a):
    for k in range(10000000):
        a[k] += 1

# Função principal do script
def main():

    # Criando um array de 1.000.000.000 (1 bilhão) de elementos preenchidos com 1
    a = np.ones(1000000000, dtype=np.float64)

    # Executando a função FillArrayWithCPU e medindo o tempo de execução
    times_to_run = 10
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        FillArrayWithCPU(a)
        toc = perf_counter_ns()
        timing[i] = toc - tic
    timing *= 1e-6
    print(f"\nTempo de Execução na CPU (Sequencial): {timing.mean():.3f} ms")

    # Executando a função FillArrayWithGPU e medindo o tempo de execução
    times_to_run = 10
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        cuda.synchronize()
        tic = perf_counter_ns()
        FillArrayWithGPU(a)
        cuda.synchronize()
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"\nTempo de Execução na GPU: {timing.mean():.3f} ms") 
 
    # Execução Paralela
    times_to_run = 10
    timing = np.empty(times_to_run, dtype=np.float32)
    for i in range(timing.size):
        tic = perf_counter_ns()
        FillArrayWithJITParallel(a)
        toc = perf_counter_ns()
        timing[i] = toc-tic
    timing *= 1e-6
    print(f"\nTempo de Execução de Forma Paralela em Múltiplos Cores da CPU: {timing.mean():.3f} ms") 

    print("\nFim\n")

if __name__ == "__main__" :
    main()

# dsa_script-02.py
# Projeto 1 - Multiplicação de Vetores com Vetorização e Paralelização em CPU
# Execução na CPU com Vetorização e Paralelismo

# Importando o módulo numpy como np para operações com arrays
import numpy as np

# Importando a função default_timer do módulo timeit e renomeando-a como timer
from timeit import default_timer as timer

# Importando a função vectorize do pacote numba para otimização
from numba import vectorize

# Definindo uma função vetorizada que multiplica dois elementos
# @vectorize é um decorador fornecido pela biblioteca Numba. 
# O propósito de um decorador é modificar ou estender o comportamento da função que ele decora. 
# Neste caso, @vectorize é usado para transformar uma função Python simples em uma função vetorizada.
# target='cpu' indica que a função será executada na CPU do computador. Numba também suporta outros targets, como 'cuda' para execução em GPUs NVIDIA.
#@vectorize(["float32(float32, float32)"], target='cpu')

# Comente a linha anterior e descomente a linha abaixo
@vectorize(["float32(float32, float32)"], target='parallel')
def dsa_multiplica_vetorial_arrays(a, b):

    # Retorna a multiplicação dos elementos a e b
    return a * b

# Definindo a função principal do script
def main():

    # Definindo o tamanho dos arrays como 100.000.000
    N = 100000000

    # Criando três arrays: A e B preenchidos com uns, C preenchido com zeros, todos do tipo float32
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)

    # Marcando o tempo inicial
    start = timer()
    
    # Chamando a função dsa_multiplica_vetorial_arrays com os arrays A e B, e armazenando o resultado em C
    C = dsa_multiplica_vetorial_arrays(A, B)
    
    # Calculando o tempo gasto para a operação de multiplicação
    tempo_total = timer() - start
    
    # Imprimindo o tempo total gasto na multiplicação
    print("\n")
    print("Oi DSA. A multiplicação vetorial dos arrays levou %f segundos" % tempo_total)
    print("\n")

# Verificando se o script é o módulo principal e executando a função main
if __name__ == '__main__':
    main()

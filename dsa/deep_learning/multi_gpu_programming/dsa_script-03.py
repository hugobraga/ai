# dsa_script-03.py
# Projeto 1 - Multiplicação de Vetores com Paralelização em GPU
# Execução na GPU com Vetorização

# Importando o módulo numpy como np para operações com arrays
import numpy as np

# Importando a função default_timer do módulo timeit e renomeando-a como timer
from timeit import default_timer as timer

# Importando a função vectorize e o módulo cuda do pacote numba para otimização
from numba import vectorize, cuda

# Definindo uma função vetorizada que multiplica dois elementos
# https://numba.pydata.org/numba-doc/latest/user/vectorize.html
@vectorize(["float32(float32, float32)"], target='cuda')
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
    print("Oi DSA. A multiplicação vetorial dos arrays com GPU levou %f segundos" % tempo_total)
    print("\n")

# Verificando se o script é o módulo principal e executando a função main
if __name__ == '__main__':
    main()


"""
A diferença de desempenho entre as duas operações vetorizadas usando Numba, uma com target='cuda' e a outra com target='parallel', 
pode ser atribuída a várias razões. Vamos analisar os possíveis motivos:

Overhead de Uso de GPU (CUDA): Quando você usa target='cuda', a operação é projetada para ser executada em uma GPU NVIDIA. 
Embora as GPUs sejam extremamente rápidas para certos tipos de operações paralelas, especialmente aquelas envolvendo grandes conjuntos de dados, 
elas também têm um overhead significativo para comunicação e inicialização. Se a quantidade de dados não for suficientemente grande para justificar 
o overhead de usar a GPU, a operação pode acabar sendo mais lenta do que a execução paralela na CPU.

Paralelismo na CPU: Com target='parallel', a função é executada em paralelo na CPU. Modernas CPUs são capazes de executar várias threads em paralelo 
e para tarefas que não são extremamente intensivas em dados ou complexas, o paralelismo na CPU pode ser mais eficiente do que offloading para a GPU, 
especialmente considerando o overhead de transferir dados para e da GPU.

Tamanho e Complexidade dos Dados: O desempenho relativo de operações na CPU e GPU depende muito do tamanho e da complexidade dos dados sendo processados. 
Para operações mais simples ou conjuntos de dados menores, a CPU pode ser mais rápida devido ao menor overhead. GPUs são mais eficientes para operações 
complexas e grandes conjuntos de dados.

Configuração do Hardware: O desempenho de target='cuda' depende fortemente do tipo e da capacidade da GPU. Se a GPU não for poderosa o suficiente, ou 
se a configuração do sistema não estiver otimizada para operações em CUDA, o desempenho pode ser inferior ao esperado.

Latência de Transferência de Dados: Há uma latência associada à transferência de dados entre a memória da CPU e da GPU. Se o tempo gasto nessa 
transferência for maior que o ganho de desempenho da computação na GPU, isso pode resultar em um desempenho geral mais lento.
"""


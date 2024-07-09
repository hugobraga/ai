# dsa_script-01.py
# Projeto 1 - Multiplicação de Vetores Usando CPU e Sem Otimização
# Execução na CPU sem Vetorização

# Importando o módulo numpy como np para operações matemáticas e manipulação de arrays
import numpy as np

# Importando a função default_timer do módulo timeit e renomeando-a como timer
from timeit import default_timer as timer

# Definindo a função dsa_multiplica_arrays que multiplica elementos correspondentes de dois arrays
def dsa_multiplica_arrays(a, b, c):
    
    # Percorrendo cada índice do array 'a'
    for i in range(a.size):
        
        # Multiplicando elementos correspondentes dos arrays 'a' e 'b', e armazenando em 'c'
        c[i] = a[i] * b[i]

# Definindo a função principal do script
def main():

    # Definindo o tamanho dos arrays como 100.000.000
    N = 100000000

    # Criando três arrays: A e B preenchidos com uns, C preenchido com zeros, todos do tipo float32
    A = np.ones(N, dtype = np.float32)
    B = np.ones(N, dtype = np.float32)
    C = np.zeros(N, dtype = np.float32)

    # Marcando o tempo inicial
    start = timer()
    
    # Chamando a função dsa_multiplica_arrays com os arrays A, B e C
    dsa_multiplica_arrays(A, B, C)
    
    # Calculando o tempo gasto para a operação de multiplicação
    tempo_total = timer() - start
    
    # Imprimindo o tempo total gasto na multiplicação
    print("\n")
    print("Oi DSA. A multiplicação dos arrays levou %f segundos" % tempo_total)
    print("\n")
    
# Verificando se o script é o módulo principal e executando a função main
if __name__ == '__main__':
    main()

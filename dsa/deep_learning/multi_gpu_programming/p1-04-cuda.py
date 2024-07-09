# Projeto 1
# Filtro Sobel Paralelizado na GPU

# Imports
import math  
import numpy as np 
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  
from time import perf_counter_ns 
from numba import cuda, void, float32, uint8, int32  

# Define a função de filtro Sobel para execução em GPU
@cuda.jit(void(float32[:,:], uint8[:, :], int32[:, :], int32[:, :]))
def sobel_filter(input_image, output_image, kernel_x, kernel_y):

    # Obtém as coordenadas do thread na grid CUDA
    i, j = cuda.grid(2) 

    # Armazena as dimensões da imagem
    height, width = input_image.shape  

    # Armazena as dimensões dos kernels
    kernel_height, kernel_width = kernel_x.shape  

    # Calcula as margens para evitar problemas nas bordas da imagem
    margin_y, margin_x = kernel_height // 2, kernel_width // 2

    # Aplica o filtro Sobel apenas nos pixels que não estão na margem
    if (margin_y <= i < height - margin_y) and (margin_x <= j < width - margin_x):

        # Inicializa os gradientes em X e Y
        Gx, Gy = 0, 0  

        # Aplica o filtro Sobel dinamicamente com base no tamanho do kernel
        for u in range(-margin_y, margin_y + 1):
            for v in range(-margin_x, margin_x + 1):
                img_val = input_image[i + u, j + v]
                Gx += kernel_x[u + margin_y, v + margin_x] * img_val
                Gy += kernel_y[u + margin_y, v + margin_x] * img_val

        # Calcula a magnitude do gradiente
        output_image[i, j] = min(255, max(0, math.sqrt(Gx**2 + Gy**2)))

# Define a função principal
def main():
    print("Executando Filtro Sobel na GPU")  

    # Caminho da imagem a ser processada
    image_path = 'imagem.jpeg' 

    # Lê a imagem do caminho especificado
    input_image = mpimg.imread(image_path)  

    # Converte a imagem para escala de cinza se estiver em cores
    if input_image.ndim == 3 and input_image.shape[2] == 3:
        input_image = np.dot(input_image[..., :3], [0.2989, 0.5870, 0.1140])

    # Converte a imagem para float32
    input_image = input_image.astype(np.float32)  

    # Define o tamanho do bloco e da grid para a execução do kernel CUDA
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(input_image.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(input_image.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Define os kernels de Sobel
    SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    # Coloca os kernels na memória do dispositivo (GPU) para acesso rápido
    d_sobel_x = cuda.to_device(SOBEL_X)
    d_sobel_y = cuda.to_device(SOBEL_Y)

    # Define o nome da imagem de saída
    filtered_image = None  

    # Define uma função auxiliar para aplicar o filtro Sobel
    def sobel_filter_helper():

        # Acessa a variável filtered_image do escopo externo
        nonlocal filtered_image  

        # Copia a imagem de entrada para a GPU
        input_image_device = cuda.to_device(input_image)  

        # Aloca espaço para a imagem resultante na GPU
        result_device = cuda.device_array(input_image.shape, np.uint8)  

        # Executa o kernel do filtro Sobel
        sobel_filter[blockspergrid, threadsperblock](input_image_device, result_device, d_sobel_x, d_sobel_y)

        # Sincroniza a GPU com a CPU
        cuda.synchronize() 

        # Copia o resultado de volta para a CPU
        filtered_image = result_device.copy_to_host()  

    # Número de vezes para executar o filtro
    times_to_run = 5  

    # Array para armazenar os tempos de execução
    timing = np.empty(times_to_run, dtype=np.float32)  

    # Mede o tempo de execução do filtro
    for i in range(timing.size):
        cuda.synchronize() 
        tic = perf_counter_ns() 
        sobel_filter_helper()  
        toc = perf_counter_ns()  
        timing[i] = toc - tic  

    # Converte o tempo para segundos
    timing *= 1e-9  

    print(f"Tempo Médio de Execução: {timing.mean():.3f} segundos")  

    del d_sobel_x  # Libera a memória do kernel X na GPU
    del d_sobel_y  # Libera a memória do kernel Y na GPU

    # Configura e exibe o resultado do filtro
    plt.rcParams["figure.figsize"] = (8, 8)  
    fig, ax = plt.subplots()  
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  
    plt.imshow(filtered_image, cmap='gray') 
    plt.title('Resultado') 
    plt.axis('off') 
    plt.savefig('imagem-p1-04.png', bbox_inches='tight', pad_inches=0) 

# Executa a função principal
if __name__ == '__main__':
    main() 


# No fim, tudo se resume a operações com matrizes! :-)



# Projeto 1
# Filtro Sobel Otimizado com Numba e JIT

# Imports
import numpy as np 
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg 
from numba import jit, prange, uint8, float32 
from time import perf_counter_ns  

# Decorador JIT com paralelismo para otimização de função via CPU
@jit(uint8[:, :](float32[:, :]), nopython=True, parallel=True)
def sobel_filter_parallel(input_image):

    # Define os kernels do filtro Sobel
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Inicializa os arrays para os gradientes X e Y
    sobel_x = np.zeros_like(input_image)
    sobel_y = np.zeros_like(input_image)

    # Aplica o filtro Sobel
    for i in prange(1, input_image.shape[0] - 1):
        for j in prange(1, input_image.shape[1] - 1):
            sobel_x[i, j] = np.sum(Gx * input_image[i-1:i+2, j-1:j+2])
            sobel_y[i, j] = np.sum(Gy * input_image[i-1:i+2, j-1:j+2])

    # Calcula a magnitude do gradiente
    magnitude = np.hypot(sobel_x, sobel_y)

    # Limita os valores no intervalo [0, 255]
    magnitude = np.clip(magnitude, 0, 255) 

    # Converte para uint8
    return magnitude.astype(np.uint8)  

# Função principal
def main():

    print("Executando Filtro Sobel com Numba JIT")  

    # Caminho da imagem a ser processada
    image_path = 'imagem.jpeg'  

    # Carrega a imagem
    input_image = mpimg.imread(image_path)  

    # Converte a imagem para escala de cinza se necessário
    if input_image.ndim == 3 and input_image.shape[2] == 3:
        input_image = np.dot(input_image[..., :3], [0.2989, 0.5870, 0.1140])

    # Converte a imagem para float32
    input_image = input_image.astype(np.float32)  

    # Inicializa a variável para a imagem filtrada
    filtered_image = None  

    # Função auxiliar para aplicar o filtro Sobel
    def sobel_filter_helper():

        # Acessa a variável do escopo externo
        nonlocal filtered_image  

        # Aplica o filtro Sobel
        filtered_image = sobel_filter_parallel(input_image)  

     # Número de execuções para medição do tempo
    times_to_run = 5 

    # Array para armazenar os tempos
    timing = np.empty(times_to_run, dtype=np.float32)  

    for i in range(timing.size):
        tic = perf_counter_ns()  
        sobel_filter_helper()  
        toc = perf_counter_ns() 
        timing[i] = toc-tic  

    # Converte o tempo para segundos
    timing *= 1e-9  
    
    print(f"Tempo Médio de Execução: {timing.mean():.3f} segundos")  # Imprime o tempo médio

    # Configura e exibe o resultado do filtro
    plt.rcParams["figure.figsize"] = (8, 8)  # Define o tamanho da figura
    fig, ax = plt.subplots()  # Cria uma figura e um subplot
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Ajusta os limites da subplot
    plt.imshow(filtered_image, cmap='gray')  # Exibe a imagem filtrada em escala de cinza
    plt.title('Resultado')  # Define o título da figura
    plt.axis('off')  # Desativa os eixos
    plt.savefig('imagem-p1-03.png', bbox_inches='tight', pad_inches=0)  # Salva a figura

# Executa a função main se o script for o principal
if __name__ == '__main__':
    main()  # Executa a função principal

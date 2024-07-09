# Projeto 1
# Filtro Sobel com Programação Python

# Imports
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  
from time import perf_counter_ns  

# Define uma função para aplicar o filtro Sobel em uma imagem
def sobel_filter(input_image):

    # Define o kernel do filtro Sobel para a direção X
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  

    # Define o kernel do filtro Sobel para a direção Y
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  

    # Inicializa matrizes para os gradientes X e Y
    grad_x = np.zeros_like(input_image)
    grad_y = np.zeros_like(input_image)

    # Aplica o filtro Sobel na imagem
    for i in range(1, input_image.shape[0] - 1):

        for j in range(1, input_image.shape[1] - 1):

            # Aplica Gx e calcula o gradiente em X
            grad_x[i, j] = np.sum(Gx * input_image[i-1:i+2, j-1:j+2])  

            # Aplica Gy e calcula o gradiente em Y
            grad_y[i, j] = np.sum(Gy * input_image[i-1:i+2, j-1:j+2])  

    # Calcula a magnitude do gradiente
    magnitude = np.hypot(grad_x, grad_y)  

    # Limita os valores da magnitude entre 0 e 255
    magnitude = np.clip(magnitude, 0, 255)  

    # Converte a magnitude para uint8
    return magnitude.astype(np.uint8)  

# Define a função principal do script
def main():

    print("Executando Filtro Sobel com Programação Python") 

    # Define o caminho para a imagem de entrada
    image_path = 'imagem.jpeg'  

    # Lê a imagem do caminho especificado
    input_image = mpimg.imread(image_path)  

    # Verifica se a imagem está em cores e a converte para escala de cinza
    if input_image.ndim == 3 and input_image.shape[2] == 3:

        # Converte para escala de cinza
        input_image = np.dot(input_image[..., :3], [0.2989, 0.5870, 0.1140]) 

    # Converte a imagem para float32
    input_image = input_image.astype(np.float32) 

    # Inicializa a variável para armazenar a imagem filtrada
    filtered_image = None  

    # Define uma função auxiliar para aplicar o filtro Sobel
    def dsa_executa_sobel_filter():

        # Indica que filtered_image é uma variável do escopo externo
        nonlocal filtered_image  

        # Aplica o filtro Sobel
        filtered_image = sobel_filter(input_image)  

     # Define o número de vezes que o filtro será aplicado para medição de tempo
    times_to_run = 5

    # Cria um array para armazenar os tempos de execução
    timing = np.empty(times_to_run, dtype=np.float32)  

    # Loop para executar o filtro várias vezes
    for i in range(timing.size):  

        # Marca o tempo de início
        tic = perf_counter_ns()  

        # Aplica o filtro Sobel
        dsa_executa_sobel_filter()  

        # Marca o tempo de fim
        toc = perf_counter_ns()  

        # Calcula o tempo de execução
        timing[i] = toc - tic  

    # Converte o tempo de execução para segundos
    timing *= 1e-9  

    # Imprime o tempo médio de execução
    print(f"Tempo Médio de Execução:: {timing.mean():.3f} segundos") 

    # Configura e exibe o resultado do filtro
    plt.rcParams["figure.figsize"] = (8, 8)  
    fig, ax = plt.subplots()  
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1) 
    plt.imshow(filtered_image, cmap='gray')  
    plt.title('Resultado')  
    plt.axis('off')  
    plt.savefig('imagem-p1-02.png', bbox_inches='tight', pad_inches=0) 

# Executa a função main
if __name__ == '__main__':
    main()  

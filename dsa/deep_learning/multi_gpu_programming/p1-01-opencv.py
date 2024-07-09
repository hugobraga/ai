# Projeto 1
# Filtro Sobel com OpenCV

# pip install opencv-python

# Imports
import cv2  
import matplotlib.pyplot as plt  
import numpy as np  
from time import perf_counter_ns  

# Definindo a função de filtro Sobel
def sobel_filter(image_gray):
    
    # Aplicar o filtro Sobel na direção X
    # Sobel em X com kernel 3x3
    sobel_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)  

    # Aplicar o filtro Sobel na direção Y
    # Sobel em Y com kernel 3x3
    sobel_y = cv2.Sobel(image_gray, cv2.CV_32F, 1, 1, ksize=3)  

    # Calcular a magnitude dos gradientes
    magnitude = np.hypot(sobel_x, sobel_y) 

    # Clip (limite) dos valores para o intervalo 0-255
    magnitude = np.clip(magnitude, 0, 255)  

    # Converter a magnitude para uint8 e retornar
    return magnitude.astype(np.uint8) 

# Definindo a função principal
def main():

    print("Executando o Filtro Sobel com OpenCV")

    # Caminho da imagem a ser lida
    image_path = 'imagem.jpeg' 

    # Lendo a imagem em cores
    image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)  

    # Convertendo a imagem para escala de cinza
    image_grayscale = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY) 

    # Inicializando a variável para a imagem filtrada
    filtered_image = None

    # Definindo uma função auxiliar para aplicar o filtro Sobel
    def dsa_executa_sobel_filter():

        # Acessando a variável filtered_image do escopo externo
        nonlocal filtered_image  

        # Aplicando o filtro Sobel
        filtered_image = sobel_filter(image_grayscale)  

    # Definindo o número de vezes que o filtro será executado
    times_to_run = 5

    # Criando um array para armazenar os tempos de execução
    timing = np.empty(times_to_run, dtype=np.float32)

    # Executando o filtro várias vezes e medindo o tempo
    for i in range(timing.size):
        tic = perf_counter_ns() 
        dsa_executa_sobel_filter() 
        toc = perf_counter_ns() 
        timing[i] = toc - tic  

    # Convertendo o tempo para segundos
    timing *= 1e-9

    # Imprimindo o tempo médio de execução
    print(f"Tempo Médio de Execução: {timing.mean():.3f} segundos")

    # Configurando e exibindo o resultado do filtro
    plt.rcParams["figure.figsize"] = (8, 8) 
    fig, ax = plt.subplots()  
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)  
    plt.imshow(filtered_image, cmap='gray')  
    plt.title('Resultado')  
    plt.axis('off') 
    plt.savefig('imagem-p1-01.png', bbox_inches='tight', pad_inches=0) 

# Executando a função principal se o script for o principal executado
if __name__ == '__main__':
    main()

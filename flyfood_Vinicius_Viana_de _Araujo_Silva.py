import time
import matplotlib.pyplot as plt
import numpy as np

tempo_inicial = time.time()

matriz = open("matriz.txt")
conteudo = matriz.readlines()


def obter_pontos(conteudo):
    points = []
    for indice_da_linha in range(len(conteudo)):
        linha = conteudo[indice_da_linha].split()
        for index_da_coluna in range(len(linha)):
            if linha[index_da_coluna] != '0':
                points.append((linha[index_da_coluna], indice_da_linha, index_da_coluna))
    return points

def calcular_grafo(pontos):
    grafo = {}
    # inicializar dicionario com listas
    for ponto in pontos:
        vertice = ponto[0]
        grafo[vertice] = []
        # grafo vai ser algo como {R -> [], A -> []...}

    for ponto in pontos:
        for ponto_adjacente in pontos:
            if ponto != ponto_adjacente:
                distancia = calcular_distancia(ponto, ponto_adjacente)
                verticeOrigem = ponto[0]
                verticeDestino = ponto_adjacente[0]
                grafo[verticeOrigem].append((verticeDestino, distancia))
    return grafo

def calcular_distancia(ponto_x, ponto_y):
    return abs(ponto_x[1] - ponto_y[1]) + abs(ponto_x[2] - ponto_y[2])

def distancia_vertice(vertice_procurado, lista_adjacencia):
    for adjacencia in lista_adjacencia:
        vertice_adjacente, distancia = adjacencia
        if vertice_adjacente == vertice_procurado:
            return distancia

def permutar(conjunto):
    if len(conjunto) == 1:
        return [conjunto]
    todas_as_permutacoes = []
    for indice, elemento in enumerate(conjunto):
        restante = conjunto[:indice] + conjunto[indice + 1:]
        permutacoes_restante = permutar(restante)
        elemento_e_permutacoes = [[elemento] + p for p in permutacoes_restante]
        todas_as_permutacoes.extend(elemento_e_permutacoes)

    return todas_as_permutacoes

def todas_as_rotas(grafo):
    vertices = []
    for vertice in grafo.keys():
        if vertice != "R":
            vertices.append(vertice)

    rotas = []
    for permutacao in permutar(vertices):
        primeiro_vertice = permutacao[0]
        distancia_inicial = distancia_vertice(primeiro_vertice, grafo['R'])
        ultimo_vertice = permutacao[-1]
        distancia_final = distancia_vertice('R', grafo[ultimo_vertice])

        distancia_miolo_rota = 0
        for i in range(len(permutacao) - 1):
            vertice_inicio = permutacao[i]
            vertice_destino = permutacao[i + 1]
            distancia_miolo_rota += distancia_vertice(vertice_destino, grafo[vertice_inicio])

        distancia_total = distancia_inicial + distancia_miolo_rota + distancia_final
        rotas.append((distancia_total, permutacao))

    return rotas

pontos = obter_pontos(conteudo)
grafo = calcular_grafo(pontos)
rotas = todas_as_rotas(grafo)
distancia_muitominima = 0

for distancia_minima in range(len(rotas)):
    if rotas [distancia_minima] < rotas[distancia_muitominima]:
        distancia_muitominima = distancia_minima

print(rotas[distancia_muitominima])
'''
tempo_final = time.time()
tempo_de_execução = tempo_final - tempo_inicial
print(f'O tempo de execução foi {tempo_de_execução} segundos.')
'''
'''
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y = [0.0009996891021728516, 0.0009992122650146484, 0.0009992122650146484, 0.0009999275207519531, 0.0010008811950683594, 0.013992547988891602, 0.05496835708618164, 0.3667902946472168, 6.353364706039429, 54.365203619003296, 8290.716258764267]

plt.xlabel('Número de pontos de entregas')
plt.ylabel('Tempo de execução em segundos')

plt.title('Gráfico pontos de entrega x tempo de execução')

for i in range(0, len(x)):
    plt.plot(x[i:i+2], y[i:i+2], 'ks-')

plt.xticks(x)
plt.show()

'''
'''
melhor_percurso = np.array([[3, 0], [1, 1], [0, 4], [2, 4], [3, 2], [3, 0]])
labels = ['R', 'Ponto A', 'Ponto D', 'Ponto C', 'Ponto B', 'R']


for i in range(len(labels)):
    label = labels[i]
    if i < (len(labels) - 1):
        plt.plot(
            np.array([melhor_percurso[i: i + 1, 0], melhor_percurso[i: i + 1, 0]]),
            np.array(
                [melhor_percurso[i: i + 1, 1], melhor_percurso[i + 1: i + 2, 1]]
            ),
            "bd-",
            linewidth=2,
            markersize=15,
        )
        plt.plot(
            np.array(
                [melhor_percurso[i: i + 1, 0], melhor_percurso[i + 1: i + 2, 0]]
            ),
            np.array(
                [melhor_percurso[i + 1: i + 2, 1], melhor_percurso[i + 1: i + 2, 1]]
            ),
            "bd-",
            linewidth=2,
            markersize=15,
        )
        plt.annotate(
            label,  # this is the text
            (
                melhor_percurso[i: i + 1, 0],
                melhor_percurso[i: i + 1, 1],
            ),  # these are the coordinates to position the label
            textcoords="offset points",  # how to position the text
            xytext=(0, 10),  # distance from text to point (x,y)
            ha="center",
            fontsize=15,
        )  # horizontal alignment can be left, right or center
    else:
        plt.annotate(
            label,  # this is the text
            (
                melhor_percurso[i: i + 1, 0],
                melhor_percurso[i: i + 1, 1],
            ),  # these are the coordinates to position the label
            textcoords="offset points",  # how to position the text
            xytext=(0, 10),  # distance from text to point (x,y)
            ha="center",
            fontsize=14,
        )  # horizontal alignment can be left, right or center
plt.title(f"Melhor percurso baseado na matriz de {len(labels)-2} instâncias", fontsize=20)
plt.xlabel("Coordenada X", fontsize=20)
plt.ylabel("Coordenada Y", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=15)
plt.show()
print(melhor_percurso)


'''





from sklearn import datasets
import numpy as np
import copy
import matplotlib.pyplot as plt


def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))


def sigmoidDerivada(sig):
    return sig * (1 - sig)


base = datasets.load_breast_cancer()
entradas = base.data

# Normalizacao das entradas
# xr = (xr – min(xR)) / (max(xR) – min(xR))
# yr = (yr – min(yR)) / (max(yR) – min(yR))
#evita que os pesos sejam ajustados desproporcionamente.

for i in range(30):
    a = entradas[:, i]
    entradas[:, i] = (a-np.amin(a))/(np.amax(a)-np.amin(a))


# Deslocamento dos valores de entrada 0(minimo) e 1(maximo) em 0.05
#reduz o custo computacional quando os pesos sinapticos tendem a infinito 
entradas[entradas == 1] = 0.95
entradas[entradas == 0] = 0.05


saida = base.target
dados = np.insert(entradas, 30, saida, axis=1)
# distribuicao aleatoria dos dados da base de dados

np.random.shuffle(dados)
bias = 1
treino = copy.deepcopy(dados[142:])
treino = np.insert(treino, 0, bias, axis=1)

# replica de saidas ==0
#balanceamento das amostras já que o número de amostras malignas é menor.

replica = treino[treino[:, 31] == 0]
treino = np.concatenate((treino, replica))
# distribuicao aleatoria dos dados de treino
np.random.shuffle(treino)


entradas_Treino = treino[:, :31]
saidas_Treino = treino[:, 31:]


teste = copy.deepcopy(dados[0:142])
teste = np.insert(teste, 0, bias, axis=1)
replica = teste[teste[:, 31] == 0]
teste = np.concatenate((teste, replica))
np.random.shuffle(teste)
entradas_Teste = teste[:, :31]
saidas_Teste = teste[:, 31:]


epocas = 10000
taxaAprendizagem = 0.001
momentum = 1
camadaSaida = 0
num_camadasOcultas = 10
pesos0 = 0.6 * np.random.rand(31, num_camadasOcultas) - 0.3
pesos1 = 0.6 * np.random.rand(num_camadasOcultas + 1, 1) - 0.3


listErroSaida_Treino = list()
listErroSaida_Teste = list()
listEpocas = list()


listErroClassificacao_Treino = list()
listErroClassificacao_Teste = list()


for j in range(epocas):
    camadaEntrada = entradas_Treino
    # multiplicacao de matriz dos valores de entrada pelos respectivos pesos
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    # ativação dos valores de cada neuronio da camada oculta
    camadaOculta = sigmoid(somaSinapse0)
    camadaOculta = np.insert(camadaOculta, num_camadasOcultas, bias, axis=1)

    # multiplicacao de matriz de cada neuronio da camada de saída com os pesos da camada de saída
    somaSinapse1 = np.dot(camadaOculta, pesos1)

    # funcao de ativacao aplicada aos valores das sinapses
    camadaSaida = sigmoid(somaSinapse1)
    
    # calculo do erro quadrático da época
    #penaliza erros maiores e reduz o treinamento em erros menores.
   erroCamadaSaida = (saidas_Treino - camadaSaida)
    sinal = np.where(erroCamadaSaida > 0, 1, -1)
    erroCamadaSaida = (erroCamadaSaida**2)*sinal

    erroSaida_Treino = sum(abs(erroCamadaSaida))
    listErroSaida_Treino.append(erroSaida_Treino)
    listEpocas.append(j)

    erroClassificacao_Treino = abs(sum(np.around(camadaSaida) - saidas_Treino))
    listErroClassificacao_Treino.append(int(erroClassificacao_Treino[0]))

    # derivada parcial do valor da camada de saída
    derivadaSaida = sigmoidDerivada(camadaSaida)
    # calculo do delta da camada de saída erro * derivada da funcao de ativacao
    deltaSaida = erroCamadaSaida * derivadaSaida

    pesos1Transposta = pesos1.T
    # calculo do delta da camada oculta
    # deltaEscondida = derivadaSigmoide*peso*deltaSaida
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    deltaCamadaOculta = np.delete(
        deltaCamadaOculta, num_camadasOcultas, axis=1)

    # backpropagation
    # peso = (pesos*momento)+(entrada * delta * taxaAprendizagem)
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momentum) + (pesosNovo1 * taxaAprendizagem)

    camadaEntradaTransposta = camadaEntrada.T
    pesosnovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momentum) + (pesosnovo0 * taxaAprendizagem)

    # FeedFoward Teste
    camadaEntrada_Teste = entradas_Teste
    somaSinapse0_Teste = np.dot(camadaEntrada_Teste, pesos0)

    camadaOculta_Teste = sigmoid(somaSinapse0_Teste)
    camadaOculta_Teste = np.insert(
        camadaOculta_Teste, num_camadasOcultas, bias, axis=1)
    somaSinapse1_Teste = np.dot(camadaOculta_Teste, pesos1)

    camadaSaida_Teste = sigmoid(somaSinapse1_Teste)
    erroSaida_Teste = sum(abs(saidas_Teste - camadaSaida_Teste))

    listErroSaida_Teste.append(erroSaida_Teste)

    print(f'Epoca: {j}', end=' ')
    print(f'Erro  Aproximação: {erroSaida_Treino}', end='|')
    print(f'Erro  Teste: {erroSaida_Teste}', end='|')

    erroClassificacao_Teste = abs(
        sum(np.around(camadaSaida_Teste) - saidas_Teste))
    listErroClassificacao_Teste.append(int(erroClassificacao_Teste[0]))
    print(f'Erro Classificação treino:{erroClassificacao_Treino[0]}', end='|')
    print(f'Erro Classificação teste:{erroClassificacao_Teste[0]}', end='|')
    print()

plt.plot(listEpocas, listErroSaida_Treino)
plt.plot(listEpocas, listErroSaida_Teste)
plt.show()

plt.plot(listEpocas, listErroClassificacao_Treino)
plt.plot(listEpocas, listErroClassificacao_Teste)
plt.show()

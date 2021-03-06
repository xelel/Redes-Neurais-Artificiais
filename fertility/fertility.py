import numpy as np
import copy
import matplotlib.pyplot as plt
from random import shuffle

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))


def sigmoidDerivada(sig):
    return sig * (1 - sig)


database = np.loadtxt('fertility_Dia.txt', delimiter=',')
bias = 1
shuffle(database)

entradas = copy.deepcopy(database[:, :9])
entradas = np.insert(entradas, 0, bias, axis=1)
entradas_Treino=entradas[0:75]
entradas_Teste=entradas[75:]

saidas = copy.deepcopy(database[:, 9:])
saidas_Treino=saidas[:75]
saidas_Teste=saidas[75:]

epocas = 30000
momentum = 1
taxaAprendizagem = 0.08
camadaSaida = 0

pesos0 = 0.6*np.random.rand(10, 5)-0.3 
pesos1 = 0.6*np.random.rand(6, 1)-0.3 

listErroSaida_Treino = list()
listErroSaida_Teste = list()
listEpocas = list()

listErroClassificacao_Treino=list()
listErroClassificacao_Teste=list()
for j in range(epocas):
    camadaEntrada = entradas_Treino
    # multiplicacao de matriz dos valores de entrada pelos respectivos pesos
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    # ativação dos valores de cada neuronio da camada oculta
    camadaOculta = sigmoid(somaSinapse0)
    camadaOculta = np.insert(camadaOculta, 5, bias, axis=1)

    # camadaOculta = np.append(camadaOculta,1)
    # multiplicacao de matriz de cada neuronio da camada de saída com os pesos da camada de saída
    somaSinapse1 = np.dot(camadaOculta, pesos1)

    # funcao de ativacao aplicada aos valores das sinapses
    camadaSaida = sigmoid(somaSinapse1)
    # calculo do erro da época
    erroCamadaSaida = (saidas_Treino - camadaSaida)
    erroSaida_Treino = sum(abs(saidas_Treino - camadaSaida))
   
    listErroSaida_Treino.append(erroSaida_Treino)
    listEpocas.append(j)
    
    
    
    print(f'Treino: ')
    print(f'Epoca: {j}', end=' ')
    print(f'Erro  Aproximação: {erroSaida_Treino}', end=' ')
    
    erroClassificacao_Treino = abs(sum(np.around(camadaSaida) - saidas_Treino))
    listErroClassificacao_Treino.append(int(erroClassificacao_Treino[0]))
    print(f'Erro Classificação:{erroClassificacao_Treino[0]}')
    

    
    # derivada parcial do valor da camada de saída
    derivadaSaida = sigmoidDerivada(camadaSaida)
    # calculo do delta da camada de saída erro * derivada da funcao de ativacao
    deltaSaida = erroCamadaSaida * derivadaSaida

    pesos1Transposta = pesos1.T
    # calculo do delta da camada oculta
    # deltaEscondida = derivadaSigmoide*peso*deltaSaida
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    deltaCamadaOculta = np.delete(deltaCamadaOculta, 5, axis=1)

    # backpropagation
    # peso = (pesos*momento)+(entrada * delta * taxaAprendizagem)
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momentum) + (pesosNovo1 * taxaAprendizagem)

    camadaEntradaTransposta = camadaEntrada.T
    pesosnovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momentum) + (pesosnovo0 * taxaAprendizagem)

    #FeedFoward Teste
    camadaEntrada_Teste=entradas_Teste
    somaSinapse0_Teste=np.dot(camadaEntrada_Teste,pesos0)
    
    camadaOculta_Teste=sigmoid(somaSinapse0_Teste)
    camadaOculta_Teste = np.insert(camadaOculta_Teste, 5, bias, axis=1)
    somaSinapse1_Teste=np.dot(camadaOculta_Teste,pesos1)
    
    camadaSaida_Teste=sigmoid(somaSinapse1_Teste)    
    erroSaida_Teste = sum(abs(saidas_Teste - camadaSaida_Teste))
    listErroSaida_Teste.append(erroSaida_Teste)
    
    
    
    
    
    print(f'Teste: ')
    print(f'Epoca: {j}', end=' ')
    print(f'Erro  Teste: {erroSaida_Teste}', end=' ')
    
    erroClassificacao_Teste = abs(sum(np.around(camadaSaida_Teste) - saidas_Teste))
    listErroClassificacao_Teste.append(int(erroClassificacao_Teste[0]))
    print(f'Erro Classificação:{erroClassificacao_Teste[0]}')
    print('=-' * 15)

plt.plot(listEpocas,listErroSaida_Treino)
plt.plot(listEpocas,listErroSaida_Teste)
plt.show()

plt.plot(listEpocas,listErroClassificacao_Treino)
plt.plot(listEpocas,listErroClassificacao_Teste)
plt.show()


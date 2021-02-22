import numpy as np


def sigmoid(soma):
    return 1/(1+np.exp(-soma))


def sigmoidDerivada(sig):
    return sig * (1-sig)


entradas = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])

saidas = np.array([[0], [0], [0], [1]])

#Pesos Camada Entrada para camada Oculta
pesos0 = 2*np.random.random((2, 3))-1
#Pesos camada Oculta para camadaSaida
pesos1 = 2*np.random.random((3, 1))-1

epocas = 7000
momentum = 1
taxaAprendizagem = 0.3
camadaSaida = 0


for j in range(epocas):
    camadaEntrada = entradas
    # multiplicacao de matriz dos valores de entrada pelos respectivos pesos
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    # ativação dos valores de cada neuronio da camada oculta
    camadaOculta = sigmoid(somaSinapse0)
    # multiplicacao de matriz de cada neuronio da camada de saída com os pesos da camada de saída
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    # funcao de ativacao aplicada aos valores das sinapses
    camadaSaida = sigmoid(somaSinapse1)
    # calculo do erro da época
    erroCamadaSaida = (saidas - camadaSaida)
    erroSaida = sum(abs(saidas - camadaSaida))
    
    print('=-'*15)
    print(f'Epoca: {j}',end=' ')
    print(f'Erro saída: {erroSaida}',end=' ')
    erroClassificacao =  abs(sum(np.around(camadaSaida) - saidas))
    print(f'Erro Classificação:{erroClassificacao}')
    print('=-'*15)
    
    #print(f'Erro {mediaAbsoluta}')
    # derivada parcial do valor da camada de saída
    derivadaSaida = sigmoidDerivada(camadaSaida)
    # calculo do delta da camada de saída erro * derivada da funcao de ativacao
    deltaSaida = erroCamadaSaida * derivadaSaida

    pesos1Transposta = pesos1.T
    # calculo do delta da camada oculta
    #deltaEscondida = derivadaSigmoide*peso*deltaSaida
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso*sigmoidDerivada(camadaOculta)

    # backpropagation
    #peso = (pesos*momento)+(entrada * delta * taxaAprendizagem)
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momentum) + (pesosNovo1*taxaAprendizagem)

    camadaEntradaTransposta = camadaEntrada.T
    pesosnovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0*momentum) + (pesosnovo0*taxaAprendizagem)


erroClassificacao =  abs(sum(np.around(camadaSaida) - saidas))
print(f'Erro Classificação:{erroClassificacao}')

    
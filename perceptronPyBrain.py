
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
# cria-se um conjunto de entradas (dataset) para treinamento
# São passadas as dimensões dos vetores de entrada e do objetivo


def treinamento_Portas(list_Entrada_Saida, NumCamadasOcultas, taxa_aprendizado, epochs):
    # adiciona-se as amostras
    d_in = 0
    d_out = 0
    for d in list_Entrada_Saida:
        d_in = len(d[0])
        d_out = len(d[1])

    dataset = SupervisedDataSet(d_in, d_out)
    for l in list_Entrada_Saida:
        entrada = l[0]
        saida = l[1]
        dataset.addSample(entrada, saida)

    # construindo a rede

    network = buildNetwork(dataset.indim, NumCamadasOcultas,
                           dataset.outdim, bias=True, hiddenclass=SigmoidLayer, outclass=SigmoidLayer,)

    # utilizando o backpropagation
    trainer = BackpropTrainer(
        network, dataset, learningrate=taxa_aprendizado)

    # trainamento da rede
    for epocas in range(epochs):
        trainer.train()

    # teste da rede
    test_data = SupervisedDataSet(d_in, d_out)
    for l in list_Entrada_Saida:
        entrada = l[0]
        saida = l[1]
        test_data.addSample(entrada, saida)

    try:
        trainer.testOnData(test_data, verbose=True)
    except:
        pass


if __name__ == "__main__":
    print('\033[31mPorta Xor:\033[m')

    entrada_saida_xor = [[[1, 1], [0]], [[1, 0], [1]],
                         [[0, 1], [1]], [[0, 0], [0]]]

    rede = treinamento_Portas(list_Entrada_Saida=entrada_saida_xor,
                              NumCamadasOcultas=1, taxa_aprendizado=0.3, epochs=7000)

    print('=-'*15)
    print('\033[31mPorta And:\033[m')
    entrada_saida_and = [[[1, 1], [1]], [[1, 0], [0]],
                         [[0, 1], [0]], [[0, 0], [0]]]

    rede = treinamento_Portas(list_Entrada_Saida=entrada_saida_and,
                              NumCamadasOcultas=4, taxa_aprendizado=0.3, epochs=1000)

    print('=-'*15)
    print('\033[31mPorta Or:\033[m')
    entrada_saida_or = [[[1, 1], [1]], [[1, 0], [1]],
                        [[0, 1], [1]], [[0, 0], [0]]]

    rede = treinamento_Portas(list_Entrada_Saida=entrada_saida_or,
                              NumCamadasOcultas=2, taxa_aprendizado=0.2, epochs=1000)
    print('=-'*15)

    print('\033[31mTreinamento Robô:\033[m')
    entrada_saida_robo = [[[0, 1, 1], [0, 1]], [[1, 0, 0], [1, 0]]]

    rede = treinamento_Portas(list_Entrada_Saida=entrada_saida_robo,
                              NumCamadasOcultas=2, taxa_aprendizado=0.2, epochs=1000)

from pybrain.datasets import dataset
from pybrain.structure.modules.biasunit import BiasUnit
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import ReluLayer
from copy import deepcopy
import numpy as np

database = np.loadtxt('fertility_Dia.txt', delimiter=',')
rede = buildNetwork(9, 6, 1, hiddenclass=ReluLayer,
                    bias=False, outclass=SigmoidLayer)
# print(rede['in'])
# print(rede['hidden0'])
print(rede['out'])
# print(rede['bias'])
entradas = deepcopy(database[:, :9])
saidas = deepcopy(database[:, 9:])

base = SupervisedDataSet(9, 1)

for l in range(len(entradas)):
    base.addSample(entradas[l], saidas[l])


treinamento = BackpropTrainer(
    rede, dataset=base, learningrate=0.06, momentum=0.3, verbose=True)

for i in range(1, 10000):
    erro = treinamento.train()

    #print(f'Erro época {i}: {erro}')

# print(rede.activate[])

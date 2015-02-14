import sys
from nn import *


def create_network(params):
    if params.arch == 1:
        # Number of features extracted by the conv net
        x = (params.height - params.kw0 + 1)//params.pool0
        x = (x - params.kw1 + 1)//params.pool1
        x = (x - params.kw2 + 1)//params.pool2
        x = x*x*params.nhu2
        # Network creation
        layers = []
        layers.append(ConvPool(3,
                               params.nhu0,
                               params.kw0,
                               params.kw0,
                               params.pool0,
                               params.pool0))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu0,
                               params.nhu1,
                               params.kw1,
                               params.kw1,
                               params.pool1,
                               params.pool1))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu1,
                               params.nhu2,
                               params.kw2,
                               params.kw2,
                               params.pool2,
                               params.pool2))
        layers.append(ReLU())
        layers.append(Flatten())
        layers.append(Linear(x, params.nhu3))
        layers.append(ReLU())
        layers.append(Linear(params.nhu3, params.nhu4))
        layers.append(ReLU())
        layers.append(Linear(params.nhu4, 2))
        layers.append(Softmax())
        crit = NLLCriterion()
        network = Net(layers, crit, params.momentum_factor, 
                      params.L1_factor, params.L2_factor)
        return network
    else:
        print 'Unknown architecture!'
        sys.exit()

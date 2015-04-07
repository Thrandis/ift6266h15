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
        layers.append(ConvPool(3, params.nhu0,
                               params.kw0, params.kw0,
                               params.pool0, params.pool0,
                               w_init=None, b_init=0.01)) #0.01
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu0, params.nhu1,
                               params.kw1, params.kw1,
                               params.pool1, params.pool1,
                               w_init=None, b_init=0.01))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu1, params.nhu2,
                               params.kw2, params.kw2,
                               params.pool2, params.pool2,
                               w_init=None, b_init=0.01))
        layers.append(ReLU())
        layers.append(Flatten())
        layers.append(Linear(x, params.nhu3, w_init=None, b_init=0.01))
        layers.append(ReLU())
        layers.append(Linear(params.nhu3, params.nhu4, w_init=None, b_init=0.01))
        layers.append(ReLU())
        layers.append(Linear(params.nhu4, 2, w_init=None, b_init=0.0))
        layers.append(Softmax())
        crit = NLLCriterion()
        network = Net(layers, crit, params)
        return network
    elif params.arch == 2: 
        # Network creation
        layers = []
        layers.append(ConvPool(3, params.nhu0,
                               params.kw0, params.kw0,
                               params.pool0, params.pool0,
                               w_init=None, b_init=0.0)) #0.01
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu0, params.nhu1,
                               params.kw1, params.kw1,
                               params.pool1, params.pool1,
                               w_init=None, b_init=0.0))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu1, params.nhu2,
                               params.kw2, params.kw2,
                               params.pool2, params.pool2,
                               w_init=None, b_init=0.0))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu2, params.nhu3,
                               params.kw3, params.kw3,
                               params.pool3, params.pool3,
                               w_init=None, b_init=0.0))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu3, params.nhu4,
                               params.kw4, params.kw4,
                               params.pool4, params.pool4,
                               w_init=None, b_init=0.0))
        layers.append(ReLU())
        layers.append(Flatten())
        layers.append(Linear(params.nhu4, params.nhu5, 
                             w_init=None, b_init=0.0))
        layers.append(ReLU())
        layers.append(Linear(params.nhu5, params.nhu6, w_init=None, b_init=0.0))
        layers.append(ReLU())
        layers.append(Linear(params.nhu6, 2, w_init=None, b_init=0.0))
        layers.append(Softmax())
        crit = NLLCriterion()
        network = Net(layers, crit, params)
        return network
    elif params.arch == 3: 
        layers = []
        if params.grey:
            x = 1
        else:
            x = 3
        layers.append(ConvPool(x, params.nhu0,
                               params.kw0, params.kw0,
                               params.pool0, params.pool0,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout)) #0.01
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu0, params.nhu1,
                               params.kw1, params.kw1,
                               params.pool1, params.pool1,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu1, params.nhu2,
                               params.kw2, params.kw2,
                               params.pool2, params.pool2,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu2, params.nhu3,
                               params.kw3, params.kw3,
                               params.pool3, params.pool3,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(ReLU())
        layers.append(Flatten())
        layers.append(Linear(params.nhu3, params.nhu4, w_init=None, b_init=0.01,
                             dropout=params.dropout))
        layers.append(ReLU())
        layers.append(Linear(params.nhu4, 2, w_init=None, b_init=0.0,
                             dropout=params.dropout))
        layers.append(Softmax())
        crit = NLLCriterion()
        network = Net(layers, crit, params)
        return network
    elif params.arch == 4: 
        layers = []
        c1 = ConvPool(5, params.nhu0, 8, 8, 2, 2,
                      w_init=None, b_init=0.0)
        c2 = ConvPool(params.nhu0, params.nhu1, 8, 8, 1, 1, 
                      w_init=None, b_init=0.0)
        c3 = ConvPool(params.nhu1, 2, 1, 1, 1, 1,
                      w_init=None, b_init=0.0)
        layers.append(c1)
        layers.append(HardTanh())
        layers.append(c2)
        layers.append(HardTanh())
        layers.append(c3)
        layers.append(HardTanh())
        crit = NLLCriterion()
        network = PedroNet(layers, crit, params)
        return network
    elif params.arch == 5: 
        """Same as arch 3 but with PReLU"""
        layers = []
        if params.grey:
            x = 1
        else:
            x = 3
        layers.append(ConvPool(x, params.nhu0,
                               params.kw0, params.kw0,
                               params.pool0, params.pool0,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout)) #0.01
        layers.append(PReLU(params.nhu0))
        layers.append(ConvPool(params.nhu0, params.nhu1,
                               params.kw1, params.kw1,
                               params.pool1, params.pool1,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(PReLU(params.nhu1))
        layers.append(ConvPool(params.nhu1, params.nhu2,
                               params.kw2, params.kw2,
                               params.pool2, params.pool2,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(PReLU(params.nhu2))
        layers.append(ConvPool(params.nhu2, params.nhu3,
                               params.kw3, params.kw3,
                               params.pool3, params.pool3,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(PReLU(params.nhu3))
        layers.append(Flatten())
        layers.append(Linear(params.nhu3, params.nhu4, w_init=None, b_init=0.01,
                             dropout=params.dropout))
        layers.append(PReLU(params.nhu4))
        layers.append(Linear(params.nhu4, 2, w_init=None, b_init=0.0,
                             dropout=params.dropout))
        layers.append(Softmax())
        crit = NLLCriterion()
        network = Net(layers, crit, params)
        return network
    elif params.arch == 6:
        """Same as arch 3 with batch norm""" 
        layers = []
        if params.grey:
            x = 1
        else:
            x = 3
        layers.append(ConvPool(x, params.nhu0,
                               params.kw0, params.kw0,
                               params.pool0, params.pool0,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout)) #0.01
        layers.append(ReLU())
        layers.append(BatchNorm(params.nhu0))
        layers.append(ConvPool(params.nhu0, params.nhu1,
                               params.kw1, params.kw1,
                               params.pool1, params.pool1,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(ReLU())
        layers.append(BatchNorm(params.nhu1))
        layers.append(ConvPool(params.nhu1, params.nhu2,
                               params.kw2, params.kw2,
                               params.pool2, params.pool2,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(ReLU())
        layers.append(BatchNorm(params.nhu2))
        layers.append(ConvPool(params.nhu2, params.nhu3,
                               params.kw3, params.kw3,
                               params.pool3, params.pool3,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(ReLU())
        layers.append(BatchNorm(params.nhu3))
        layers.append(Flatten())
        layers.append(Linear(params.nhu3, params.nhu4, w_init=None, b_init=0.01,
                             dropout=params.dropout))
        layers.append(ReLU())
        layers.append(BatchNorm(params.nhu4))
        layers.append(Linear(params.nhu4, 2, w_init=None, b_init=0.0,
                             dropout=params.dropout))
        layers.append(Softmax())
        crit = NLLCriterion()
        network = Net(layers, crit, params)
        return network
    elif params.arch == 7:
        """Same as arch 3 with batch norm""" 
        layers = []
        if params.grey:
            x = 1
        else:
            x = 3
        layers.append(ConvPool(x, params.nhu0,
                               params.kw0, params.kw0,
                               params.pool0, params.pool0,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout)) #0.01
        layers.append(BatchNorm(params.nhu0))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu0, params.nhu1,
                               params.kw1, params.kw1,
                               params.pool1, params.pool1,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(BatchNorm(params.nhu1))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu1, params.nhu2,
                               params.kw2, params.kw2,
                               params.pool2, params.pool2,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(BatchNorm(params.nhu2))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu2, params.nhu3,
                               params.kw3, params.kw3,
                               params.pool3, params.pool3,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(BatchNorm(params.nhu3))
        layers.append(ReLU())
        layers.append(Flatten())
        layers.append(Linear(params.nhu3, params.nhu4, w_init=None, b_init=0.01,
                             dropout=params.dropout))
        layers.append(BatchNorm(params.nhu4))
        layers.append(ReLU())
        layers.append(Linear(params.nhu4, 2, w_init=None, b_init=0.0,
                             dropout=params.dropout))
        layers.append(Softmax())
        crit = NLLCriterion()
        network = Net(layers, crit, params)
        return network
    elif params.arch == 8:
        """Same as 3 with double conv""" 
        layers = []
        if params.grey:
            x = 1
        else:
            x = 3
        layers.append(ConvPool(x, params.nhu0,
                               params.kw0, params.kw0,
                               1, 1,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout)) #0.01
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu0, params.nhu0,
                               params.kw0, params.kw0,
                               params.pool0, params.pool0,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout)) #0.01
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu0, params.nhu1,
                               params.kw1, params.kw1,
                               1, 1,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu1, params.nhu1,
                               params.kw1, params.kw1,
                               params.pool1, params.pool1,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu1, params.nhu2,
                               params.kw2, params.kw2,
                               1, 1,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu2, params.nhu2,
                               params.kw2, params.kw2,
                               params.pool2, params.pool2,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu2, params.nhu3,
                               params.kw3, params.kw3,
                               1, 1,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(ReLU())
        layers.append(ConvPool(params.nhu3, params.nhu3,
                               params.kw3, params.kw3,
                               params.pool3, params.pool3,
                               w_init=None, b_init=0.01,
                               dropout=params.dropout))
        layers.append(ReLU())
        layers.append(Flatten())
        layers.append(Linear(params.nhu3, params.nhu4, w_init=None, b_init=0.01,
                             dropout=params.dropout))
        layers.append(ReLU())
        layers.append(Linear(params.nhu4, 2, w_init=None, b_init=0.0,
                             dropout=params.dropout))
        layers.append(Softmax())
        crit = NLLCriterion()
        network = Net(layers, crit, params)
        return network
    elif params.arch == 99: # For Alex: To compare with Pylearn2
        layers = []
        layers.append(ConvPool(3, 32, 4, 4, 2, 2, w_init=0.1))
        layers.append(ReLU())
        layers.append(ConvPool(32, 16, 4, 4, 2, 2, w_init=0.1))
        layers.append(ReLU())
        layers.append(ConvPool(16, 16, 4, 4, 2, 2, w_init=0.1))
        layers.append(ReLU())
        layers.append(ConvPool(16, 16, 4, 4, 2, 2, w_init=0.1))
        layers.append(ReLU())
        layers.append(Flatten())
        layers.append(Linear(1936, 16, w_init=1.0))
        layers.append(ReLU())
        layers.append(Linear(16, 16, w_init=1.0))
        layers.append(ReLU())
        layers.append(Linear(16, 2, w_init=1.0))
        layers.append(Softmax())
        crit = NLLCriterion()
        network = Net(layers, crit, params)
        return network
    else:
        print 'Unknown architecture!'
        sys.exit()

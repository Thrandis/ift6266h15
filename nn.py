from numpy.random import randn
import numpy as np
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox import cuda
import theano.tensor as T
import theano


theano.config.floatX = 'float32'
floatX = theano.config.floatX

# TODO: Check Borrow


# ----------------------------------------------------------------------------
# Layers


class Layer():

    def fwd(self, x):
        pass

    def get_weights(self):
        return None

    def get_bias(self):
        return None


class Linear(Layer):

    def __init__(self, in_size, out_size):
        scale = np.sqrt(6./(in_size + out_size)) 
        W_val = scale*randn(in_size, out_size).astype(floatX)
        self.W = theano.shared(name='W', value=W_val)
        b_val = np.zeros(out_size, dtype=floatX)
        self.b = theano.shared(name='b', value=b_val)

    def fwd(self, x):
        return T.dot(x, self.W) + self.b
    
    def get_weights(self):
        return self.W
        
    def get_bias(self):
        return self.b


class Softmax(Layer):

    def fwd(self, x):
        return T.nnet.softmax(x)


class ReLU(Layer):
    
    def fwd(self, x):
        return T.switch(x < 0, 0, x)


class HardTanh(Layer):
    
    def fwd(self, x):
        x = T.switch(x < -1, -1, x)
        return T.switch(x > 1, 1, x)


class Flatten(Layer):

    def fwd(self, x):
        return T.flatten(x, outdim=2)


class ConvPool(Layer):

    def __init__(self, in_size, out_size, kh, kw, ph, pw):
        fan_in = in_size*kh*kw
        fan_out = out_size*kh*kw/(ph*pw)
        scale = np.sqrt(6./(fan_in + fan_out)) 
        W_val = scale*randn(out_size, in_size, kh, kw).astype(floatX)
        self.W = theano.shared(name='W', value=W_val)
        b_val = np.zeros(out_size, dtype=floatX)
        self.b = theano.shared(name='b', value=b_val)
        self.ph = ph
        self.pw = pw
    
    def fwd(self, x):
        # TODO : CHECK OTHER PARAM OF CONV2D FOR OPTIMISATION
        y = T.nnet.conv.conv2d(x, self.W)
        z = max_pool_2d(y, (self.ph, self.pw), ignore_border=True)
        return z + self.b.dimshuffle('x', 0, 'x', 'x')

    def get_weights(self):
        return self.W
        
    def get_bias(self):
        return self.b


class NLLCriterion():

    def fwd(self, x, t):
        L = -(t*T.log(x)).sum(axis=1).mean()
        M = T.neq(t.argmax(axis=1), x.argmax(axis=1)).mean() 
        return L, M


# ----------------------------------------------------------------------------
# Network


class Net():

    def __init__(self, layers, criterion, momentum_factor, 
                 L1_factor, L2_factor):
        self.layers = layers      
        self.criterion = criterion
        self.train, self.predict = self._compile_net(layers, 
                                                     criterion,
                                                     momentum_factor,
                                                     L1_factor,
                                                     L2_factor)                             
    
    def _get_parameters(self):
        weights = []
        bias = []
        for layer in self.layers:
            w = layer.get_weights()
            if w is not None:
                weights.append(w)
            b = layer.get_bias()
            if b is not None:
                bias.append(b)
        return weights, bias

    def _set_parameters(self, weights, bias):
        for layer in self.layers:
            w = layer.get_weights()
            if w is not None:
                w = weights.pop(0)
            b = layer.get_bias()
            if b is not None:
                b = bias.pop(0)
           
    def __getstate__(self):
        weights, bias = self._get_parameters()
        return (self.layers, self.criterion, weights, bias)

    def __setstate__(self, state):
        layers, criterion, weights, bias = state
        self.layers = layers
        self.criterion = criterion
        self._set_parameters(weights, bias)
        
    def _compile_net(self, layers, criterion, momentum_factor,
                     L1_factor, L2_factor):
        # Building symbolic forward
        x = T.tensor4('x')
        t = T.matrix('t')
        y = cuda.basic_ops.gpu_from_host(x)
        t = cuda.basic_ops.gpu_from_host(t)
        for layer in layers:
            y = layer.fwd(y)
        L, M = criterion.fwd(y, t)
        
        # Getting weights and bias
        lr = T.scalar('lr')
        weights, bias = self._get_parameters()
        
        # Regularization
        if L1_factor != 0:
            s = 0.
            for w in weights:
                s += abs(w).sum()
            L = L + L1_factor*s
        if L2_factor != 0:
           s = 0.
           for w in weights:
               s += (w ** 2).sum()
           L = L + 0.5*L2_factor*s
      
        # Computing gradients
        g_weights = T.grad(L, weights)
        g_bias = T.grad(L, bias)
        
        # Momentum
        if momentum_factor != 0:
            M_w = []
            for w in weights:
                m_val = np.zeros(w.get_value().shape, dtype=floatX)
                M_w.append(theano.shared(value=m_val))
            M_b = []
            for b in bias:
                m_val = np.zeros(b.get_value().shape, dtype=floatX)
                M_b.append(theano.shared(value=m_val))
            momentum_w = [momentum_factor*m - lr*g
                          for m, g in zip(M_w, g_weights)]
            momentum_b = [momentum_factor*b - lr*g
                          for b, g in zip(M_b, g_bias)]
            updates_w = [(w, w + m) for m, w in zip(momentum_w, weights)]
            updates_b = [(b, b + m) for m, b in zip(momentum_b, bias)]
            updates = updates_w + updates_b             
        else:
            # Symbolic updates
            updates_w = [(w, w - lr*g) for w, g in zip(weights, g_weights)]
            updates_b = [(b, b - lr*g) for b, g in zip(bias, g_bias)]
            updates = updates_w + updates_b
        # Compiling expressions
        train = theano.function(inputs=[theano.Param(x, borrow=True), 
                                        theano.Param(t, borrow=True),
                                        lr], 
                                outputs=[theano.Out(L, borrow=True),
                                         theano.Out(M, borrow=True)], 
                                updates=updates)
        predict = theano.function(inputs=[x, t],
                                  outputs=[L, M])
        return train, predict


# ----------------------------------------------------------------------------
# Functions


def lr_decay(lr, alpha, epoch):
    return lr/(1. + epoch*alpha)

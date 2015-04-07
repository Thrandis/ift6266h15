import sys
from numpy.random import randn
from numpy.random import uniform
import numpy as np
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox import cuda
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
import theano


theano.config.floatX = 'float32'
floatX = theano.config.floatX

srng = RandomStreams()


# ----------------------------------------------------------------------------
# Layers


class Layer(object):

    def fwd(self, x):
        raise NotImplementedError

    def get_weights(self):
        return None

    def get_bias(self):
        return None

    def get_dropout(self):
        return None 

    def inf(self, x):
        return self.fwd(x)

    def get_inf_updates(self):
        return None

    def get_other(self):
        return None


class Linear(Layer):

    def __init__(self, in_size, out_size, w_init=None, b_init=0.0,
                 dropout=False):
        if w_init == None:
            scale = np.sqrt(6./(in_size + out_size)) 
            W_val = uniform(-scale, scale, (in_size, out_size)).astype(floatX)
        elif type(w_init) == float:
            W_val = uniform(-w_init, w_init, (in_size, out_size))
            W_val = W_val.astype(floatX)
        else:
            print('Bad init scheme in class Linear')
            sys.exit()  
        self.W = theano.shared(name='W', value=W_val)
        b_val = b_init*np.ones(out_size, dtype=floatX)
        self.b = theano.shared(name='b', value=b_val)
        
        # Dropout Mask
        self.dropout = dropout
        if self.dropout:
            M_val = np.ones((512,), dtype=floatX)
            self.M = theano.shared(name='M', value=M_val)

    def fwd(self, x):
        if self.dropout:
            return T.dot(self.M * x, self.W) + self.b
        return T.dot(x, self.W) + self.b
    
    def get_weights(self):
        return self.W
        
    def get_bias(self):
        return self.b
    
    def get_dropout(self):
        return self.M


class Softmax(Layer):

    def fwd(self, x):
        return T.nnet.softmax(x)


class ReLU(Layer):
    
    def fwd(self, x):
        return T.switch(x < 0, 0, x)


class PReLU(Layer):

    def __init__(self, in_size, a_init=1.0):
        a_val = a_init*np.ones(in_size, dtype=floatX)
        self.a = theano.shared(name='a', value=a_val)
        
    def fwd(self, x):
        y = T.switch(x > 0, x, 0)
        if x.ndim == 4:
            return y + self.a.dimshuffle('x', 0, 'x', 'x')*T.switch(x < 0, x, 0)
        elif x.ndim == 2:
            return y + self.a*T.switch(x < 0, x, 0)
            
#    def get_bias(self):
#        return self.a


class HardTanh(Layer):
 
    def fwd(self, x):
        return T.clip(x, -1, 1)


class Flatten(Layer):

    def fwd(self, x):
        return T.flatten(x, outdim=2)


class BatchNorm(Layer):

    def __init__(self, in_size, epsilon=1e-6):
        self.epsilon = epsilon
        gamma_val = np.ones(in_size, dtype=floatX)
        self.gamma = theano.shared(name='gamma', value=gamma_val)
        beta_val = np.zeros(in_size, dtype=floatX)
        self.beta = theano.shared(name='beta', value=beta_val)
        means_val = np.zeros(in_size, dtype=floatX)
        self.means = theano.shared(name='means', value=means_val)
        variances_val = np.ones(in_size, dtype=floatX)
        self.variances = theano.shared(name='varainces', value=variances_val)

    def fwd(self, x):
        if x.ndim == 2:
            means = x.mean(axis=0, keepdims=True, dtype=floatX)
            variances = x.var(axis=0, keepdims=True)
            self.symb_means = x.mean(axis=0, dtype=floatX) 
            self.symb_variances = x.var(axis=0)
            x_hat = (x - means)/T.sqrt(variances + self.epsilon)
            return self.gamma*x_hat + self.beta 
        elif x.ndim == 4:
            means = x.mean(axis=[0,2,3], keepdims=True, dtype=floatX)
            variances = x.var(axis=[0,2,3], keepdims=True) 
            self.symb_means = x.mean(axis=[0,2,3], dtype=floatX)
            self.symb_variances = x.var(axis=[0,2,3])
            x_hat = (x - means)/T.sqrt(variances + self.epsilon)
            return self.gamma.dimshuffle('x', 0, 'x', 'x')*x_hat + self.beta.dimshuffle('x', 0, 'x', 'x') 
        else:
            raise NotImplementedError

    def inf(self, x):
        if x.ndim == 2:
            x_hat = (x - self.means)/T.sqrt(self.variances + self.epsilon)
            return self.gamma*x_hat + self.beta 
        elif x.ndim == 4:    
            x_hat = (x - self.means.dimshuffle('x', 0, 'x', 'x'))/T.sqrt(self.variances.dimshuffle('x', 0, 'x', 'x') + self.epsilon)
            return self.gamma.dimshuffle('x', 0, 'x', 'x')*x_hat + self.beta.dimshuffle('x', 0, 'x', 'x') 
        else:
            raise NotImplementedError
    
    def get_inf_updates(self):
        return [(self.means, self.symb_means),
                (self.variances, self.symb_variances)]

    def get_weights(self):
        return self.gamma

    def get_bias(self):
        return self.beta


class ConvPool(Layer):

    def __init__(self, in_size, out_size, kh, kw, ph, pw, 
                 w_init=None, b_init=0.0, dropout=False):
        if w_init == None:
            fan_in = in_size*kh*kw
            fan_out = out_size*kh*kw/(ph*pw)
            scale = np.sqrt(6./(fan_in + fan_out)) 
            W_val = uniform(-scale, scale, (out_size, in_size, kh, kw)).astype(floatX)
        elif type(w_init) == float:
            W_val = uniform(-w_init, w_init, (out_size, in_size, kh, kw))
            W_val = W_val.astype(floatX)
        else:
            print('Bad init scheme in class ConvPool')
            sys.exit()  
        self.W = theano.shared(name='W', value=W_val)
        
        b_val = b_init*np.ones(out_size, dtype=floatX)
        self.b = theano.shared(name='b', value=b_val)
        
        self.ph = ph
        self.pw = pw

    def fwd(self, x):
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

    def __init__(self, layers, criterion, params):
        self.layers = layers
        self.criterion = criterion
        self.train, self.predict = self._compile_net(layers, criterion, 
                                                     params)
        self.dropout = params.dropout
        self.dropout_masks = None
        if self.dropout:
            self.dropout_masks = []
            for layer in self.layers:
                m = layer.get_dropout()
                if m is not None:
                    self.dropout_masks.append(m)                             
    
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

    def _symbolic_forward(self, layers, criterion):
        x = T.tensor4('x')
        t = T.matrix('t')
        y = cuda.basic_ops.gpu_from_host(x)
        t = cuda.basic_ops.gpu_from_host(t)
        for layer in layers:
            y = layer.fwd(y)
        L, M = criterion.fwd(y, t)
        inputs = [theano.Param(x, borrow=True), 
                  theano.Param(t, borrow=True)]
        return L, M, inputs

    def _symbolic_inf(self, layers, criterion):
        x = T.tensor4('x')
        t = T.matrix('t')
        y = cuda.basic_ops.gpu_from_host(x)
        t = cuda.basic_ops.gpu_from_host(t)
        for layer in layers:
            y = layer.inf(y)
        L, M = criterion.fwd(y, t)
        inputs = [theano.Param(x, borrow=True), 
                  theano.Param(t, borrow=True)]
        return L, M, inputs
        

    def _compile_net(self, layers, criterion, params):
        L, M, inputs = self._symbolic_forward(layers, criterion)
        
        lr = params.lr
        L1_factor = params.L1_factor
        L2_factor = params.L2_factor
        momentum_factor = params.momentum_factor

         
        # Getting weights and bias
        lr = T.scalar('lr')
        weights, bias = self._get_parameters()
        
        # Regularization
        if L1_factor != 0.0:
            s = 0.
            for w in weights:
                s += abs(w).sum()
            L = L + L1_factor*s
        if L2_factor != 0.0:
           s = 0.
           for w in weights:
               s += (w ** 2).sum()
           L = L + 0.5*L2_factor*s
      
        # Computing gradients
        g_weights = T.grad(L, weights)
        g_bias = T.grad(L, bias)
        
        # Momentum
        if momentum_factor != 0.0:
            M_w = []
            for w in weights:
                m_val = np.zeros(w.get_value().shape, dtype=floatX)
                M_w.append(theano.shared(value=m_val))
            M_b = []
            for b in bias:
                m_val = np.zeros(b.get_value().shape, dtype=floatX)
                M_b.append(theano.shared(value=m_val))
            updates_m_w = [(m, momentum_factor*m - lr*g)
                           for m, g in zip(M_w, g_weights)]
            updates_m_b = [(m, momentum_factor*m - lr*g)
                           for m, g in zip(M_b, g_bias)]
            # Nesterov Momentum
            if params.NAG:
                updates_w = [(w, w + momentum_factor*m - lr*g)
                             for w, m, g in zip(weights, M_w, g_weights)]
                updates_b = [(b, b + momentum_factor*m - lr*g)
                             for b, m, g in zip(bias, M_b, g_bias)]
            # Classic Momentum
            else:
                updates_w = [(w, w + m) for m, w in zip(M_w, weights)]
                updates_b = [(b, b + m) for m, b in zip(M_b, bias)]
            updates = updates_m_w + updates_m_b + updates_w + updates_b
        else:
            updates_w = [(w, w - lr*g) for w, g in zip(weights, g_weights)]
            updates_b = [(b, b - lr*g) for b, g in zip(bias, g_bias)]
            updates = updates_w + updates_b
        # Updates for batch norm
        inf_updates = []
        for layer in layers:
            u = layer.get_inf_updates()
            if u is not None:
                inf_updates.extend(u)
        updates = updates + inf_updates
        # Updates for PReLU
        prelu_updates = []
        for layer in layers:
            u = layer.get_other()
            if u is not None:
                prelu_updates.append(u)
        g_prelu = T.grad(L, prelu_updates)
        prelu_updates = [(a, a) for a, g in zip(prelu_updates, g_prelu)]
        updates = updates + prelu_updates
        # Compiling expressions
        train = theano.function(inputs=inputs + [lr], 
                                outputs=[theano.Out(L, borrow=True),
                                         theano.Out(M, borrow=True)], 
                                updates=updates)
        # Building symbolic prediction
        L_hat, M_hat, inputs_hat = self._symbolic_inf(layers, criterion)
        # Regularization 2nd take:
        if L1_factor != 0.0:
            s = 0.
            for w in weights:
                s += abs(w).sum()
            L_hat = L_hat + L1_factor*s
        if L2_factor != 0.0:
           s = 0.
           for w in weights:
               s += (w ** 2).sum()
           L_hat = L_hat + 0.5*L2_factor*s
         
        predict = theano.function(inputs=inputs_hat,
                                  outputs=[L_hat, M_hat])
        return train, predict


class PedroNet(Net):

    def __init__(self, layers, criterion, params):
        Net.__init__(self, layers, criterion, params)

    def _symbolic_forward(self, layers, criterion):
        x1 = T.tensor4('x1')
        x2 = T.tensor4('x2')
        x3 = T.tensor4('x3')
        y1 = cuda.basic_ops.gpu_from_host(x1)
        y2 = cuda.basic_ops.gpu_from_host(x2)
        y3 = cuda.basic_ops.gpu_from_host(x3)
        t = T.matrix('t')
        t = cuda.basic_ops.gpu_from_host(t)
        for layer in layers:
            y1 = layer.fwd(y1)
        y2 = T.concatenate([y2, y1], axis=1)
        for layer in layers:
            y2 = layer.fwd(y2)
        y3 = T.concatenate([y3, y2], axis=1)
        for layer in layers[0:-1]:
            y3 = layer.fwd(y3)
        y3 = Flatten().fwd(y3)
        y3 = Softmax().fwd(y3)
        L, M = criterion.fwd(y3, t)
        inputs = [theano.Param(x1, borrow=True),
                  theano.Param(x2, borrow=True),
                  theano.Param(x3, borrow=True),
                  theano.Param(t, borrow=True)]
        return L, M, inputs


# ----------------------------------------------------------------------------
# Functions


def lr_decay(lr, alpha, epoch):
    return lr/(1. + epoch*alpha)


def drop(network, action, p, p0):
    masks = network.dropout_masks
    if action == 'train':
        for i in xrange(len(masks)):
            s = masks[i].get_value().shape
            v = np.random.randint(0, 101, s).astype(floatX)
            if i == 0:
                v[np.where(v < 100*p0)] = 1
            else:
                v[np.where(v < 100*p)] = 1
            v[np.where(v != 1)] = 0
            masks[i].set_value(v)
    elif action == 'valid':
        for i in xrange(len(masks)):
            s = masks[i].get_value().shape
            if i == 0:
                v = (1./p0)*np.ones(s, dtype=floatX)
            else:
                v = (1./p)*np.ones(s, dtype=floatX)
            masks[i].set_value(v)
    else:
        print 'Unknown action in drop'
        sys.exit()


# ----------------------------------------------------------------------------
# Unit Testing

if __name__ == '__main__':
     
    x = T.tensor4('x')
    L = BatchNorm(3)
    y = L.fwd(x)
    f = theano.function([x], y)
    
    x = randn(1, 3, 100, 100)
    for i in range(1, 5):
        xx = i + i*randn(1, 3, 100, 100)
        x = np.vstack([x, xx])
    x = x.astype(floatX)
    
    print x.shape
    print x.mean(axis=0).mean(axis=1).mean(axis=1)
    print x.var(axis=0).var(axis=1).var(axis=1)
    
    L.get_weights().set_value(np.array([100,-100,200], dtype=floatX))

    y = f(x)
    print '-------'
    print y.shape
    print y.mean(axis=0).mean(axis=1).mean(axis=1) 
    print y.var(axis=0).var(axis=1).var(axis=1)
    
    print '------------------------------------------------------------------'    

    x = T.matrix('x')
    L = BatchNorm(2)
    y = L.fwd(x)
    f = theano.function([x], y)
    
    x = randn(1, 2)
    for i in range(1, 5):
        xx = i + i*randn(1, 2)
        x = np.vstack([x, xx])
    x = x.astype(floatX)
    
    L.get_bias().set_value(np.array([100,-200], dtype=floatX))
    
    print x.shape
    print x.mean(axis=0)
    print x.var(axis=0)

    y = f(x)
    print '-------'
    print y.shape
    print y.mean(axis=0)
    print y.var(axis=0)

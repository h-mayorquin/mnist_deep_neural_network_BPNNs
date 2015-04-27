import theano
import numpy as np
import theano.test as T

# Initialize with 0 the weight W as a matrix of shape (n_in, n_out)
n_in = 3
n_out = 2

W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                  name='W', borrow=True)
b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX),
                  name='b', borrow=True)



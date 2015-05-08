import theano.tensor as T
from theano import function
from theano import Param
from theano import shared


state = shared(0)
inc = T.iscalar('inc')  # short integer o 32 bit integer
accumulator = function([inc], state, updates=[(state, state + inc)])

so, this is the one im used to.. 

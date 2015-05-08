import numpy as np
import theano.tensor as T
from theano import function
from theano import pp

# Basic functionality
x = T.dscalar('x')  # creates a variable of doubles
y = T.dscalar('y')
z = x + y
w = x * y
f = function([x, y], [z, w])

m1 = T.dmatrix('matrix1')
m2 = T.dmatrix('m2')

z = m1 + m2
f = function([m1, m2], z)

value1 = np.zeros((2, 2))
value2 = np.arange(4).reshape((2, 2))

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = function([x], s)
print "You can pretty print tensor instances"
print pp(s)
print "You can print the output of functions as array"
print logistic(np.arange(4).reshape((2, 2)))

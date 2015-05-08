import theano.tensor as T
from theano import function
from theano import Param

x, y = T.dscalars('x', 'y')
z = x + y
f = function([x, Param(y, default=1)], z)  # Allow us to use defaults
print f(33)

x, y, w = T.dscalars('x', 'y', 'w')
z = x + y + w
f = function([x, Param(y, default=1),
              Param(w, default=2, name='w_by_name')], z)


print f(3, w_by_name=3)

from retrodiff import Dag, Node, Function

class Mul(Function):
    def forward(self, x, y): return x * y
    def backward(self, grad, wrt, x, y): return (y, x)[wrt] * grad

class Add(Function):
    def forward(self, x, y): return x + y
    def backward(self, grad, wrt, x, y): return grad

mul, add = Mul(), Add()
Node.__add__ = lambda x, y: add(x, y)
Node.__mul__ = lambda x, y: mul(x, y)

x, y, z = (Node(), Node(), Node())
f = (x * y) + z # same as: f = add(mul(x, y), z)

dag = Dag([x, y, z], f)

out = dag.forward([1, 2, 3])
grads = dag.backward(1)

print(out, grads)


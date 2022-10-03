from retrodiff import Node, Function


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

x.value = 2
y.value = 3
z.value = 1

f.forward()
f.backward(1)

out = f.value
grads = x.grad, y.grad, z.grad, f.grad

print(f.show_tree())
print('output:', out)
print('gradients:', grads)

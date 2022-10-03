# Retrodiff

Standalone reverse autodiff library.

## Usage

### Dag api

1.  Subclass `retrodiff.Function` to create base functions (or use presets from `retrodiff.utils.function`).
2.  Make the dag function by composing `Node`s and `Function`s.
3.  Set values on input nodes.
4.  Call `node.forward()` and `node.backward()` on the output node to calulate output values and gradients.
5.  Gradients and valuesa are stored in `node.grad` and in `node.value`.
    You can also see the full function with `node.show_tree()`.

Example:
```python
class Mul(Function):
    def forward(self, x, y): return x * y
    def backward(self, grad, wrt, x, y): return (y, x)[wrt] * grad

class Add(Function):
    def forward(self, x, y): return x + y
    def backward(self, grad, wrt, x, y): return grad

mul, add = Mul(), Add()
x, y, z = (Node(), Node(), Node())
f = add(mul(x, y), z)

x.value = 2
y.value = 3
z.value = 1

f.forward()
f.backward(1)

out = f.value
grads = x.grad, y.grad, z.grad, f.grad
```

See also the [examples](examples).

### Models api

See `retrodiff.utils.nn`.

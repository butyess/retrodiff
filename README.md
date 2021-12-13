# Retrodiff

Standalone reverse autodiff library.

## Usage

1.  Subclass `retrodiff.Function` to create base functions (or use presets from `retrodiff.utils`).
2.  Make the dag function by composing `Node`s and `Function`s.
3.  Create a `Dag` by passing the input nodes and the function you just made.
4.  Run `Dag.forward(*inputs)` and `Dag.backward()` to get output values and gradients.

Example:
```python
class Mul(Function):
    def forward(self, x, y): return x * y
    def backward(self, grad, wrt, x, y): return (y, x)[wrt] * grad

class Add(Function):
    def forward(self, x, y): return x + y
    def backward(self, grad, wrt, x, y): return grad

mul, add = Mul(), Add()
input_nodes = [Node(), Node(), Node()]
f = add(mul(input_nodes[0], input_nodes[1]), input_nodes[2])

dag = Dag(input_nodes, f)

out = dag.forward([1, 2, 3])
grads = dag.backward(1)
```

See also [examples](examples)

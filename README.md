# Retrodiff

Standalonoe reverse autodiff library.

## Usage

1.  Subclass `retrodiff.autodiff.Function` to create your custom functions and instantiate them.
2.  Instantiate input `Node`s objects.
3.  Create a `Dag` passing the input nodes and the function you want.
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

dag = Dag(input_nodes, add(mul(input_nodes[0], input_nodes[1]), input_nodes[2]))

out = dag.forward([1, 2, 3])
grads = dag.backward(1)
```

See also [examples](examples)

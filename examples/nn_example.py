import logging

import numpy as np
import matplotlib.pyplot as plt

from retrodiff import Dag, Function, Node
from retrodiff.nn import NeuralNetwork


logging.basicConfig(format="%(message)s", level=logging.INFO)


class Mul(Function):
    def forward(self, x, y): return x * y
    def backward(self, grad, wrt, x, y): return grad * (y, x)[wrt]

class Add(Function):
    def forward(self, *values): return sum(values)
    def backward(self, grad, wrt, *values): return grad

class Sub(Function):
    def forward(self, x, y): return x - y
    def backward(self, grad, wrt, x, y): return grad * (1, -1)[wrt]

class Square(Function):
    def forward(self, x): return x ** 2
    def backward(self, grad, wrt, x): return grad * 2 * x


class Neuron(NeuralNetwork):
    def __init__(self):
        super().__init__()

        p = [Node(), Node(), Node(), Node()] # parameters
        mul, add, square = Mul(), Add(), Square()

        self._nn_dag = Dag(p, add(mul(square(p[0]), p[1]), mul(p[0], p[2]), p[3]))
        self.weights = [0, 0, 0]


def mse_loss():
    pred, label = Node(), Node()
    square, sub = Square(), Sub()
    return Dag([pred, label], square(sub(pred, label)))


def main():
    model = Neuron()
    model.set_loss(mse_loss())

    xs = np.random.uniform(low=-10, high=10, size=(100,))
    ys = 2 * xs ** 2 + 3 * xs + 4
    model.train(0.0001, 10, xs, ys)

    tests = np.linspace(-5, 5)
    vals = 2 * tests ** 2 + 3 * tests + 4
    out = [model.evaluate(t) for t in tests]

    fig, ax = plt.subplots()
    ax.plot(tests, vals, 'r', label='actual value')
    ax.plot(tests, out, 'bo', label='predicted value')
    fig.legend()
    plt.show()


if __name__ == "__main__":
    main()


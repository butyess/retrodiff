import math
import logging
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt

from retrodiff import Dag, Function, Node
from retrodiff.nn import NeuralNetwork


logging.basicConfig(format="%(message)s", level=logging.INFO)


class Log(Function):
    def forward(self, x): return math.log(x)
    def backward(self, grad, wrt, x): return grad * (1/x)

class Exp(Function):
    def forward(self, x): return math.exp(x)
    def backward(self, grad, wrt, x): return grad * math.exp(x)

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

class Dot(Function):
    def forward(self, a, b): return np.dot(a, b)
    def backward(self, grad, wrt, a, b):
        if wrt == 0: return np.dot(grad, b.T)
        else: return np.dot(a.T, grad)


class Network(NeuralNetwork):
    def __init__(self, layers):
        super().__init__()

        p = [Node() for _ in layers]
        dot = Dot()

        self._nn_dag = Dag(p, dot(reduce(lambda x, y: nonlinear(dot(x, y)), p[:-1]), p[-1]))
        self.weights = [np.random.uniform(low=-4, high=4, size=dim) for dim in zip(layers[:-1], layers[1:])]


def main():
    model = Network([2, 16, 16, 16, 2])

    pred, label = Node(), Node()
    mul, add, sub, square = Mul(), Add(), Sub(), Square()
    loss = Dag([pred, label], add(*[]))


if __name__ == "__main__":
    main()


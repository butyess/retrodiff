from functools import reduce

import numpy as np

from .. import Dag, Node, Model
from . import Dot, ReLU, Add


class NN(Model):
    def __init__(self, layers, parameters=None, bias=True):
        super().__init__()

        dot = Dot()
        add = Add()
        relu = ReLU()
        step = lambda x, w, b: relu(add(dot(x, w), b))
        step_nb = lambda x, w: relu(dot(x, w))

        i = Node()
        w = [Node() for _ in layers[1:]]

        if bias:
            b = [Node() for _ in layers[1:]] if bias else []
            fun = reduce(lambda acc, x: step(acc, x[0], x[1]), zip(w[:-1], b[:-1]), i)
            fun = add(dot(fun, w[-1]), b[-1])
            self._dag = Dag([i] + w + b, fun)
        else:
            fun = reduce(lambda acc, x: step_nb(acc, x[0]), w[:-1], i)
            fun = dot(fun, w[-1])
            self._dag = Dag([i] + w, fun)

        if parameters is None:
            if bias:
                self.parameters = [np.random.normal(size=dim) for dim in zip(layers[:-1], layers[1:])] + \
                                  [np.random.normal(size=(1, n)) for n in layers[1:]]
            else:
                self.parameters = [np.random.normal(size=dim) for dim in zip(layers[:-1], layers[1:])]
        else:
            self.parameters = parameters

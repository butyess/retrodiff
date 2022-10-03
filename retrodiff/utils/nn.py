import numpy as np

from .. import Node
from .function import Dot, Add, ReLU


class Model:
    def __init__(self, mutable=False):
        self.input_nodes = []
        self.output_nodes = []
        self.parameter_nodes = []
        self._weights = []
        self.mutable = mutable

        if not mutable:
            self.build_dag()

    def build_dag(self, *args, **kwargs):
        pass

    def run(self, *values, **kwargs):
        if self.mutable:
            self.build_dag(*values, **kwargs)

        if len(values) != len(self.input_nodes):
            raise ValueError("Values don't match input node size")
        if not self._weights:
            raise ValueError("Weights not initialized yet")

        for v, n in zip(values, self.input_nodes):
            n.value = v
        for w, p in zip(self.weights, self.parameter_nodes):
            p.value = w

        for n in self.output_nodes:
            n.forward()
        return [n.value for n in self.output_nodes]

    def clear(self):
        for n in self.output_nodes:
            n.clear()

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, ws):
        self._weights = ws


class Sequential(Model):
    def __init__(self, *modules):
        self.modules = modules
        super().__init__(mutable=False)

    def build_dag(self):
        for p, n in zip(self.modules[:-1], self.modules[1:]):
            assert len(p.output_nodes) == len(n.input_nodes), \
                "Two modules in sequential are uncompatible"
            for new, old  in zip(p.output_nodes, n.input_nodes):
                for o in n.output_nodes:
                    o.replace_node(old, new)

        self.weights = [x for m in self.modules for x in m.weights]

        self.input_nodes = self.modules[0].input_nodes
        self.output_nodes = self.modules[-1].output_nodes
        self.parameter_nodes = [x for m in self.modules for x in m.parameter_nodes]


class Linear(Model):
    def __init__(self, activation_fn):
        self.activation_fn = activation_fn
        super().__init__(mutable=False)

    def build_dag(self):
        dot, add, relu = Dot(), Add(), ReLU()
        x, w, b = Node(), Node(), Node()
        f = self.activation_fn(add(dot(x, w), b))

        self.input_nodes = [x]
        self.output_nodes = [f]
        self.parameter_nodes = [w, b]


class Recurrent(Model):
    '''
    First input of `run` is the initial value.
    '''
    def __init__(self, recurrent_fn):
        if recurrent_fn.__code__.co_argcount != 3:
            raise ValueError("Given function does not match parameters."\
                "Should be three in the order: input, parameters, previous_output")
        self.recurrent_fn = recurrent_fn
        super().__init__(mutable=True)

    def build_dag(self, *values):
        w, x0 = Node(), Node()
        xs = [x0]
        y = x0
        for _ in range(len(values) - 1):
            x = Node()
            y = self.recurrent_fn(x, w, y)
            xs.append(x)

        self.input_nodes = xs
        self.output_nodes = [y]
        self.parameter_nodes = [w]

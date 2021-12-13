import logging

from functools import reduce

import numpy as np
import matplotlib.pyplot as plt

from retrodiff import Dag, Node, Model
from retrodiff.utils import Dot, ReLU, Add, MSELoss, GradientDescent


logging.basicConfig(format="%(message)s", level=logging.INFO)

dot = Dot()
add = Add()
relu = ReLU()

Node.__matmul__ = lambda x, y: dot(x, y)
Node.__add__ = lambda x, y: add(x, y)


class NN(Model):
    def __init__(self, layers, parameters=None, bias=True):
        super().__init__()

        i = Node()
        w = [Node() for _ in layers[1:]]

        if bias:
            b = [Node() for _ in layers[1:]] if bias else []
            fun = reduce(lambda acc, x: relu((acc @ x[0]) + x[1]), zip(w[:-1], b[:-1]), i) @ w[-1] + b[-1]
            self._dag = Dag([i] + w + b, fun)
        else:
            fun = reduce(lambda acc, x: relu(acc @ x), w[:-1], i) @ w[-1]
            self._dag = Dag([i] + w, fun)

        if parameters is None:
            if bias:
                self.parameters = [np.random.normal(size=dim) for dim in zip(layers[:-1], layers[1:])] + \
                                  [np.random.normal(size=(1, n)) for n in layers[1:]]
            else:
                self.parameters = [np.random.normal(size=dim) for dim in zip(layers[:-1], layers[1:])]
        else:
            self.parameters = parameters


def main():
    model = NN([1, 2, 2, 1])

    logging.info("initial parameters: " + str(model.parameters))

    model.set_loss(MSELoss())
    model.set_optim(GradientDescent(lr=0.001))

    f = lambda xs: 2 * xs ** 2

    # train
    inputs = list(np.linspace(-3, 3, 100))
    labels = [f(i) for i in inputs]
    model.train(100, inputs, labels)

    logging.info("final parameters: " + str(model.parameters))

    # test
    xs = list(np.linspace(-5, 5))
    ys = [f(x) for x in xs]
    pred = [model.evaluate(x).flatten() for x in xs]

    fig, ax = plt.subplots()
    ax.plot(xs, ys, 'r', label='expected')
    ax.plot(xs, pred, 'bo', label='predicted')
    fig.legend()
    plt.show()


if __name__ == "__main__":
    main()


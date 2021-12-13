import logging

from functools import reduce

import numpy as np
import matplotlib.pyplot as plt

from retrodiff import Dag, Node, Model
from retrodiff.utils import Dot, ReLU, Add, MSELoss, GradientDescent, NN


def main():
    # logging.basicConfig(format="%(message)s", level=logging.DEBUG)
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


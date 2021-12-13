import math
import logging
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from retrodiff.utils import GradientDescent, SVMBinaryLoss, NN


def plot(model, inputs, labels):
    h = 0.25
    x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    input_mesh = np.c_[np.ravel(xx), np.ravel(yy)]
    scores = model.evaluate(input_mesh)

    Z = np.argmax(scores, axis=-1).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    ax.scatter(inputs[:, 0], inputs[:, 1], c=labels, cmap=plt.cm.Spectral)
    plt.show()


def main():
    model = NN([2, 16, 16, 2])

    model.set_loss(SVMBinaryLoss())
    model.set_optim(GradientDescent(lr=0.001))

    inputs, labels = make_moons(n_samples=100, shuffle=True, noise=0.1)

    x_train = [x.reshape(1, -1) for x in inputs]
    y_train = [y.reshape(1, -1) for y in labels]

    x_test, y_test = make_moons(n_samples=100, shuffle=True, noise=0.1)
    x_test = [x.reshape(1, -1) for x in x_test]
    y_test = [y.reshape(1, -1) for y in y_test]

    for e in range(10):
        model.train(10, x_train, y_train)
        print('epoch ', e, 'avg test loss: ', model.test(x_test, y_test))

    plot(model, inputs, labels)


if __name__ == "__main__":
    main()


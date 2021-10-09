import math
import logging
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from retrodiff import Dag, Node, Function
from retrodiff.model import Model
from retrodiff.utils import Dot, GradientDescent, ReLU, MSELoss


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


class SVMBinaryLoss(Function):
    def __init__(self, margin=1):
        self.margin = margin

    def forward(self, preds, label):
        return np.maximum(0, preds[:, int(not label)] - preds[:, label] + self.margin) / preds.shape[-1]

    def backward(self, grad, wrt, preds, label):
        if wrt == 0:
            pd = np.ones(preds.shape) / 2
            pd[:, label] *= -1
            return pd
        else:
            return 0


class Network(Model):
    def __init__(self, layers):
        super().__init__()

        p = [Node() for _ in layers]
        pred, label = Node(), Node()
        dot, relu = Dot(), ReLU()
        loss = SVMBinaryLoss()

        self._dag = Dag(p, dot(reduce(lambda x, y: relu(dot(x, y)), p[:-1]), p[-1]))
        self._loss_dag = Dag([pred, label], loss(pred, label))
        self._optim = GradientDescent(lr=0.001)

        self.parameters = [np.random.normal(scale=20.0, size=dim) for dim in zip(layers[:-1], layers[1:])]


def main():
    # logging.basicConfig(format="%(message)s", level=logging.INFO)

    model = Network([2, 16, 16, 2])

    inputs, labels = make_moons(n_samples=100, shuffle=True, noise=0.1)

    cut = 75
    batch_size = 20

    x_train = [x.reshape(1, -1) for x in inputs[:cut]]
    y_train = [y.reshape(1, -1) for y in labels[:cut]]

    x_test = [x.reshape(1, -1) for x in inputs[cut:]]
    y_test = [y.reshape(1, -1) for y in labels[cut:]]

    for e in range(10):
        ri = np.random.randint(0, len(x_train), size=batch_size)
        model.train(10, [x_train[i] for i in ri], [y_train[i] for i in ri])
        print('epoch ', e, 'avg loss: ', model.test(x_test, y_test))

    pred = model.evaluate(x_test[0])
    loss = model.loss(pred, y_test[0])
    print('pred: ', pred, ' label: ', y_test[0], ' loss: ', loss)

    plot(model, inputs, labels)


if __name__ == "__main__":
    main()


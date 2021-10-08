import unittest

import numpy as np

from retrodiff import Dag, Function, Node
from retrodiff.model import Model
from retrodiff.utils import Mul, MSELoss, GradientDescent


PARAM = 10


class SimpleModel(Model):
    def __init__(self):
        super().__init__()
        self.parameters = [PARAM]
        a, x, mul = Node(), Node(), Mul()
        self._dag = Dag([a, x], mul(a, x))


class TestModel(unittest.TestCase):

    model = SimpleModel()

    def test_creation(self):
        self.assertEqual(self.model.evaluate(10), PARAM * 10)

    def test_loss(self):
        pred, out, mse = Node(), Node(), MSELoss()
        loss_dag = Dag([pred, out], mse(pred, out))
        self.model.set_loss(loss_dag)

        xs = np.random.rand(1, 10)
        ys = np.random.rand(1) * xs

        self.assertEqual(self.model.loss(xs, ys), np.sum((xs - ys)**2))

    def test_optim(self):
        self.model.set_optim(GradientDescent(lr=0.0001))

        xs = np.random.rand(100,)
        ys = PARAM / 2 * xs

        self.model.train(100, xs[:75], ys[:75])

        avg_loss = self.model.test(xs[75:], ys[75:])
        self.assertTrue(avg_loss < 10)


if __name__ == "__main__":
    unittest.main()


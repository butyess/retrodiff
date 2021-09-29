import unittest

import numpy as np

from retrodiff import Dag, Function, Node


class TestDag(unittest.TestCase):

    def test_scalar(self):
        add = Function()
        add.forward = lambda x, y: x + y
        add.backward = lambda grad, wrt, x, y: grad

        a, b = Node(), Node()
        dag = Dag([a, b], add(a, b))

        self.assertEqual(dag.forward([1, 2]), 3)
        self.assertEqual(dag.backward(1), [1, 1])

    def test_numpy(self):
        dot = Function()
        dot.forward = np.dot
        dot.backward = lambda grad, wrt, x, y: np.dot(x.T, grad) if wrt == 1 else np.dot(grad, y.T)

        a, b = Node(), Node()
        dag = Dag([a, b], dot(a, b))

        x = np.array([[2, 4, 3]])
        y = np.array([[1, 2], [3, 2], [3, 3]])
        z = np.array([[23, 21]])

        out = dag.forward([x, y])
        grads = dag.backward(np.ones(out.shape))

        self.assertTrue(np.array_equal(out, z))
        self.assertTrue(np.array_equal(grads[0], np.sum(y, axis=1).reshape(1, -1)))
        self.assertTrue(np.array_equal(grads[1], np.repeat(x, 2, axis=0).T))



if __name__ == "__main__":
    unittest.main()

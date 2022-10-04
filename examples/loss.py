from retrodiff.utils.function import MSELossFun, ReLU
from retrodiff.utils.nn import Sequential, Loss, Linear
from retrodiff.dag import Node

import numpy as np


mse = MSELossFun()
relu = ReLU()

net = Sequential(Linear(relu, np.random.randn(4, 4), np.random.randn(4)),
                 Linear(relu, np.random.randn(4, 4), np.random.randn(4)))
loss = Loss(mse, net)

l = loss.forward(np.array([1, 2, 3, 4]), np.array([0, 0, 0, 0]))
loss.backward()

print(l)
print(loss.dag.show_tree())

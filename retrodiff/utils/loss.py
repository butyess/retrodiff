import numpy as np

from ..dag import Function

class MSELoss(Function):
    def forward(self, p, y): return np.sum((p - y)**2)
    def backward(self, grad, wrt, p, y): return (1, -1)[wrt] * 2 * (y, p)[wrt]


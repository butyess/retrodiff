import numpy as np

from ..model import Optimizer

class GradientDescent(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def new_param(self, old_param, grads):
        return np.array(old_param) - np.array(grads) * self.lr


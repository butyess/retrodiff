import numpy as np

from ..model import Optimizer

class GradientDescent(Optimizer):
    def __init__(self, lr, mu=0):
        self.lr = lr
        self.mu = mu
        self.vel = None

    def new_param(self, old_param, grads):
        if self.vel is None:
            self.vel = [g for g in grads]
        else:
            self.vel = [self.mu * v + self.lr * g for v,g in zip(self.vel, grads)]

        return [old - v for old, v in zip(old_param, self.vel)]


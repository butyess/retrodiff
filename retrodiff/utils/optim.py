from .. import Optimizer


class GradientDescent(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def new_param(self, old_param, grads):
        return [old - grad * self.lr for old, grad in zip(old_param, grads)]


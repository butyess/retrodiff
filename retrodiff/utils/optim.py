class Optimizer:
    def __init__(self, model):
        self.model = model

    def step(self):
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __init__(self, model, lr):
        super().__init__(model)
        self.lr = lr

    def step(self):
        for i, p in enumerate(self.model.parameter_nodes):
            self.model.weights[i] = self.model.weights[i] - p.grad * self.lr

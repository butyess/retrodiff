from retrodiff.utils.nn import Model


class Optimizer:
    def __init__(self, model):
        self.params = model.parameter_nodes

    def step(self):
        raise NotImplementedError

import logging

from .dag import Node, Function, Dag


class Optimizer:
    def new_param(old_param, grads):
        raise NotImplementedError


class Model:
    '''
    Base class for trainable models. To create your neural network first override
    `__init__` in a subclass and define the attributes `parameters` and `_dag`.
    Then you can instantiate it and train it after setting an optimizer and a loss function.
    '''
    def __init__(self):
        self.parameters = []
        self._dag = None
        self._loss_dag = None
        self._optim = None
        pass

    def set_loss(self, loss_dag):
        '''
        Sets the loss function for the model.
        `loss_dag` output should be a scalar.
        The loss function should accept two input, in this order:
            1.  predicted values
            2.  labels
        '''
        assert isinstance(loss_dag, Dag)
        assert len(loss_dag.input_nodes) == 2
        self._loss_dag = loss_dag

    def set_optim(self, optim):
        assert isinstance(optim, Optimizer)
        self._optim = optim

    def train(self, n_iterations, inputs, labels):
        if self._loss_dag is None:
            raise ValueError("Cannot train without a loss function")

        if self._optim is None:
            raise ValueError("Cannot train without an optimizer")

        for i in range(n_iterations):
            logging.info("Iteration %d", i)
            loss_tot = 0

            for x, y in zip(inputs, labels):
                # forward
                out = self.evaluate(x)
                loss_tot += self.loss(out, y)

                # backward
                loss_grad = self._loss_dag.backward(1) # loss should return a scalar, so 1 is ok
                grads = self._dag.backward(loss_grad[0])

                # update parameters
                self.parameters = self._optim.new_param(self.parameters, grads)

            logging.info("Average loss: %d", loss_tot / len(inputs))

    def test(self, inputs, labels):
        if self._loss_dag is None:
            raise ValueError("Cannot test without a loss function")

        loss_tot = 0
        for x, y in zip(inputs, labels):
            pred = self.evaluate(x)
            loss_tot += self.loss(pred, y)

        return loss_tot / len(inputs)

    def evaluate(self, inputs):
        self._dag.clear()
        if not isinstance(inputs, list):
            inputs = [inputs]
        return self._dag.forward(inputs + self.parameters)

    def loss(self, pred, expected):
        self._loss_dag.clear()
        return self._loss_dag.forward([pred, expected])


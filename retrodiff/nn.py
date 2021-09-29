import logging

from .dag import Node, Function, Dag


class NeuralNetwork:
    '''
    Base neural network class. To create your neural network override
    `__init__` in a subclass and define the attributes `weights` and `_nn_dag`.
    Then you can instantiate your network and `evaluate` or `train` it.
    '''
    def __init__(self):
        self.weights = []
        self._nn_dag = None
        self._loss_dag = None
        pass

    def set_loss(self, loss_dag):
        '''
        Sets the loss function for the network.
        `loss_dag` output should be a scalar.
        `loss_dag` should have two input nodes, in this exact order:
            1.  predicted values
            2.  labels
        '''
        assert isinstance(loss_dag, Dag)
        assert len(loss_dag.input_nodes) == 2
        self._loss_dag = loss_dag

    def train(self, lr, n_iterations, inputs, labels):
        '''
        Trains the network with gradient descent.
        Set logging level to INFO to see progression.
        '''
        if self._loss_dag is None:
            raise ValueError("Cannot train without a loss function")

        for i in range(n_iterations):
            logging.info("Iteration %d", i)
            losses = 0
            for x, y in zip(inputs, labels):
                self._nn_dag.clear()
                self._loss_dag.clear()

                # forward
                pred = self._nn_dag.forward([x] + self.weights)
                loss = self._loss_dag.forward([pred, y])

                # backward
                loss_grad = self._loss_dag.backward(1) # loss should return a scalar, so 1 is ok
                grads = self._nn_dag.backward(loss_grad[0])

                for i, grad in enumerate(grads[1:]):
                    self.weights[i] -= grad * lr

                losses += loss
            logging.info("Average loss: %d", losses / len(inputs))

    def evaluate(self, inputs):
        '''
        Evaluate the network with the current weights.
        '''
        self._nn_dag.clear()
        if not isinstance(inputs, list):
            inputs = [inputs]
        return self._nn_dag.forward(inputs + self.weights)


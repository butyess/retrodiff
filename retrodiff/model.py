# class Optimizer:
#     def __init__(self):
#         pass
# 
#     def new_param(self, old_param, grads):
#         raise NotImplementedError
# 
# 
# class Loss:
#     def __init__(self):
#         self._dag = None
# 
#     def apply(self, pred, labels):
#         self._dag.clear()
#         return self._dag.forward([pred, labels])
# 
#     def grads(self, init_grad=1):
#         return self._dag.backward(init_grad) # loss should return a scalar, so 1 is ok


# class Model:
#     '''Base class for trainable models.
#     
#     To create your custom model first override `__init__` and
#     define `parameters` and `_dag`. Then you can instantiate
#     it and train it after setting an optimizer with `set_optim()`
#     and a loss function with `set_loss`.
#     '''
#     def __init__(self):
#         self.parameters = []
#         self._dag = None
#         self._loss = None
#         self._optim = None
# 
#     def set_loss(self, loss):
#         assert isinstance(loss, Loss)
#         self._loss = loss
# 
#     def set_optim(self, optim):
#         assert isinstance(optim, Optimizer)
#         self._optim = optim
# 
#     def evaluate(self, inputs):
#         self._dag.clear()
#         if not isinstance(inputs, list):
#             inputs = [inputs]
#         return self._dag.forward(inputs + self.parameters)
# 
#     def train(self, n_iterations, inputs, labels):
#         assert self._loss is not None, 'Loss function not defined'
#         assert self._optim is not None, 'Optimizer not defined'
# 
#         for i in range(n_iterations):
#             loss_tot = 0
# 
#             for x, y in zip(inputs, labels):
#                 # forward
#                 out = self.evaluate(x)
#                 loss_tot += self._loss.apply(out, y)
# 
#                 # backward
#                 loss_grads = self._loss.grads()
#                 grads = self._dag.backward(loss_grads[0])
# 
#                 # update parameters
#                 self.parameters = self._optim.new_param(self.parameters, grads[1:])
# 
#             logging.debug("iteration %d", i)
#             logging.debug("average loss: %d", loss_tot / len(inputs))
# 
#     def test(self, inputs, labels):
#         assert self._loss is not None
# 
#         loss_tot = 0
#         for x, y in zip(inputs, labels):
#             pred = self.evaluate(x)
#             loss_tot += self._loss.apply(pred, y)
# 
#         return loss_tot / len(inputs)
# 
# 
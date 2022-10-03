# from .. import Loss, Node, Dag
# from . import MSELossFun, SVMBinaryLossFun
# 
# 
# class MSELoss(Loss):
#     def __init__(self):
#         super().__init__()
#         pred, labl, mse = Node(), Node(), MSELossFun()
#         self._dag = Dag([pred, labl], mse(pred, labl))
# 
#     def apply(self, pred, labels):
#         return super().apply(pred, labels)
# 
#     def grads(self, init_grad=1):
#         return super().grads(init_grad)
# 
# 
# class SVMBinaryLoss(Loss):
#     def __init__(self, margin=1):
#         super().__init__()
#         pred, labl, svm = Node(), Node(), SVMBinaryLossFun(margin)
#         self._dag = Dag([pred, labl], svm(pred, labl))
# 
#     def apply(self, pred, labels):
#         return super().apply(pred, labels)
# 
#     def grads(self, init_grad=1):
#         return super().grads(init_grad)
# 
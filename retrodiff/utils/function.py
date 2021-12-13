import numpy as np

from .. import Function


class Log(Function):
    def forward(self, x): return np.log(x)
    def backward(self, grad, wrt, x): return grad * (1/x)

class Exp(Function):
    def forward(self, x): return np.exp(x)
    def backward(self, grad, wrt, x): return grad * np.exp(x)

class Mul(Function):
    def forward(self, x, y): return x * y
    def backward(self, grad, wrt, x, y): return grad * (y, x)[wrt]

class Add(Function):
    def forward(self, *values): return sum(values)
    def backward(self, grad, wrt, *values): return grad

class Sub(Function):
    def forward(self, x, y): return x - y
    def backward(self, grad, wrt, x, y): return grad * (1, -1)[wrt]

class Square(Function):
    def forward(self, x): return x ** 2
    def backward(self, grad, wrt, x): return grad * 2 * x

class Dot(Function):
    def forward(self, a, b): return np.dot(a, b)
    def backward(self, grad, wrt, a, b):
        if wrt == 0: return np.dot(grad, b.T)
        else: return np.dot(a.T, grad)

class ReLU(Function):
    def forward(self, x): return np.maximum(x, 0)
    def backward(self, grad, wrt, x): return (x > 0).astype(np.float) * grad

class MSELossFun(Function):
    def forward(self, p, y): return np.sum((p - y)**2)
    def backward(self, grad, wrt, p, y): return (1, -1)[wrt] * 2 * (p - y) * grad

class Max(Function):
    def forward(self, x, y): return np.maximum(x, y)
    def backward(self, grad, wrt, x, y):
        if x > y:
            return grad * (not wrt)
        else:
            return grad * wrt


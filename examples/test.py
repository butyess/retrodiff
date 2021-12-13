import math

from retrodiff import Node, Dag, Function
from retrodiff.utils import Log, Exp, Mul, Add

log, exp, mul, add = Log(), Exp(), Mul(), Add()

a, b, c = Node(), Node(), Node()
dag = Dag([a, b], exp(mul(a, b)))

x = 10

print(dag.forward([x, 3]))
print(dag.backward(1))
print(3 * math.exp(3 * x))


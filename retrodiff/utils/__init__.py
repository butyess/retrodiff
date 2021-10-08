import sys

try:
    import numpy
except ImportError:
    sys.exit("Numpy needs to be installed to make this module work.")

from .function import Log, Exp, Mul, Add,  Sub, Square, Dot, ReLU
from .loss import MSELoss
from .optim import GradientDescent


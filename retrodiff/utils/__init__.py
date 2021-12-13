import sys

try:
    import numpy
except ImportError:
    sys.exit("Numpy needs to be installed to make this module work.")

from .function import *
from .loss import *
from .optim import *
from .nn import *

DEBUG_MODE = True

from .basic import Basic
from .basic_conv import BasicConv
from .dummy import Dummy
from .squeezenet import Squeezenet
from .alexnet import Alexnet
from .mnasnet import Mnasnet
from .lenet5 import LeNet5

if DEBUG_MODE:
  from .elemental import Elemental

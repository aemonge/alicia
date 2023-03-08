DEBUG_MODE = True

from .basic import Basic
from .basic_conv import BasicConv
from .dummy import Dummy
from .squeezenet import SqueezeNet
from .alexnet import AlexNet
from .mnasnet import MnasNet
from .lenet5 import LeNet5

if DEBUG_MODE:
  from .elemental import Elemental

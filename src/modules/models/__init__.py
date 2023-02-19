DEBUG_MODE = True

from .basic import Basic
from .dummy import Dummy
from .squeezenet import Squeezenet
from .alexnet import Alexnet
from .mnasnet import Mnasnet

if DEBUG_MODE:
  from .elemental import Elemental

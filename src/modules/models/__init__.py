DEBUG_MODE = True

from .basic import Basic
from .dummy import Dummy
from .squeezenet import Squeezenet

if DEBUG_MODE:
  from .elemental import Elemental

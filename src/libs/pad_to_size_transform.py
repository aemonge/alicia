from dependencies.core import transforms
from dependencies.datatypes import List

class PadToSize(object):
  """
    Pad the image to the given size.

    Attributes:
    -----------
      size: Tuple|int

    Methods:
    --------
      __call__(self, x):
        x = transforms.functional.pad(x, (self.size[0] - width, self.size[1] - height, self.size[0] // 2,
  """
  def __init__(self, size: List|int):
    """
      Constructor.

      Parameters:
      -----------
        size: Tuple|int
    """
    if isinstance(size, int):
      self.size = [size, size]
    else:
      self.size = size

  def __call__(self, x):
    """
      Pad the image to the given size.

      Parameters:
      -----------
        x: PIL.Image

      Returns:
      --------
        PIL.Image
    """
    height, width = x.size
    top_pad = max(0, (self.size[1] - height + 1) // 2)
    bottom_pad = max(0, self.size[1] - height - top_pad)
    left_pad = max(0, (self.size[0] - width + 1) // 2)
    right_pad = max(0, self.size[0] - width - left_pad)

    return transforms.functional.pad(x, (left_pad, top_pad, right_pad, bottom_pad))

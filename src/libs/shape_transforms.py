import torch
from torch import Tensor


class UnShapetransform(object):
  """
    This class is used to transform a tensor from the original shape.
  """
  def __init__(self, shape:tuple):
    """
      Parameters:
      -----------
        shape: tuple
          The shape of the tensor.
    """
    self.shape = shape

  def __call__(self, tensor:Tensor, shape: tuple = (1, 28, 8)):
    """
      Transform a tensor from the original shape.

      Parameters:
      -----------
        tensor: torch.Tensor
          The tensor to be transformed.

      Returns:
      --------
        torch.Tensor
          The transformed tensor.
    """
    return torch.reshape(tensor, shape)

class Reshapetransform(object):
  """
    This class is used to transform a tensor's shape
  """
  def __init__(self, shape:tuple):
    """
      Parameters:
      -----------
        shape: tuple
          The shape of the tensor.
    """
    self.shape = shape

  def __call__(self, tensor:Tensor):
    """
      Transform a tensor's shape.

      Parameters:
      -----------
        tensor: torch.Tensor
          The tensor to be transformed.

      Returns:
      --------
        torch.Tensor
          The transformed tensor.
    """
    return torch.reshape(tensor, self.shape)
    # return image.view(image.shape[0], -1)

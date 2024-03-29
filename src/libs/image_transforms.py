from torch import Tensor
from numpy import ndarray

class ImageToMatPlotLib(object):
  """
    PyTorch tensors assume the color channel is the first dimension
      but matplotlib assumes is the third dimension
  """
  def __str__(self) -> str:
    """
      String representation.

      Returns:
      --------
        str
    """
    return f"ImageToMatPlotLib({self.shape})"

  def __init__(self, shape:tuple = (1, 2, 0)):
    """
      Parameters:
      -----------
        shape: tuple
          The shape of the tensor.
    """
    self.shape = shape

  def __call__(self, tensor:Tensor) -> ndarray:
    """
      Transform a tensor from the original shape to MatPlotLib image.

      Parameters:
      -----------
        tensor: torch.Tensor
          The tensor to be transformed.

      Returns:
      --------
        torch.Tensor
          The transformed tensor.
    """
    return tensor.numpy().transpose((1, 2, 0))

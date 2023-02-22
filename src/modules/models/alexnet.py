from dependencies.core import torch
from dependencies.datatypes import Parameter, Iterator
from .abs_module import AbsModule
from torchvision.models import AlexNet

class Alexnet(AlexNet, AbsModule):
  """
    A basic neural network module. With randomly selected features.

    Attributes:
    -----------
      labels: list
        A list of labels.
      features: list
        A list of features.
      classifier: torch.nn.Module
        A classifier.
      input_size: int
        The input size.
      num_classes: int
        The number of classes.
      dropout: float
        The dropout probability.

    Methods:
    --------
      forward(self, x: torch.Tensor) -> torch.Tensor
        A forward pass of the neural network.
      load(self, path: str) -> None
        A load of the neural network.
      save(self, path: str) -> str
        A save of the neural network.
      create(self, input_size: int, num_classes: int, dropout: float) -> None
        Re creates the neural network.
  """

  def __repr__(self):
    """
      A string representation of the neural network.

      Returns:
      --------
        : str
          A string representation 'Basic()'.
    """
    return 'Alexnet()'

  def __init__(self, *, data: dict|None = None, labels = [],
               input_size: int = 28, dropout: float = 0.0) -> None:
    """
      Constructor of the neural network.

      Parameters:
      -----------
        data: dict
          A dictionary containing the data, to load the network though the pth file.
        labels: list
          A list of labels.
        input_size: int
          The input size.
        dropout: float
          The dropout probability.
    """
    if data is None:
      if dropout > 0.0:
        AbsModule.__init__(self, labels = labels, input_size = input_size, dropout = dropout)
        AlexNet.__init__(self, num_classes=len(labels), dropout=dropout)
      else:
        AbsModule.__init__(self, labels = labels, input_size = input_size)
        AlexNet.__init__(self, num_classes=len(labels))
      self.labels = labels # AlexNet destroys the labels attribute.
    else:
      AbsModule.__init__(self, data = data)

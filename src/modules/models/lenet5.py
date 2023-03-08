from dependencies.core import torch
from dependencies.datatypes import Parameter, Iterator
from .abs_module import AbsModule

class LeNet5(AbsModule):
  """
    A basic neural network module. With randomly selected features, and Conv2d.

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
    return 'LeNet5()'

  def __init__(self, *, data: dict|None = None, labels:list = [], input_size: int = 28, dropout: float = 0.0,
               num_classes: int|None = None) -> None:
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
      AbsModule.__init__(self,
        labels = labels, input_size = input_size, dropout = dropout, num_classes = num_classes
      )
      last_size = self.__create_features__()
      self.__create_classifier__(last_size)
    else:
      AbsModule.__init__(self, data = data)

  def __create_classifier__(self, last_size: int) -> None:
    """
      Creates the classifier.

      Parameters:
      -----------
        last_size: int
          The size of the last layer.
    """
    self.classifier = torch.nn.Sequential(
      torch.nn.Linear(last_size, self.num_classes),
      torch.nn.ReLU(),
      torch.nn.LogSoftmax(dim=1)
    )

  def __create_features__(self) -> int:
    """
      Creates the features.

      Returns:
      --------
        int
          The size of the last features, to pipe it to the classifier.
    """
    last_size = 84
    self.features = torch.nn.Sequential(
      torch.nn.Conv2d(1, 6, kernel_size=5, padding=0, stride=1),
      torch.nn.AvgPool2d(kernel_size=5, stride=1),
      torch.nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1),
      torch.nn.ReLU(),
      torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=0),
      torch.nn.BatchNorm2d(16),
      torch.nn.ReLU(),

      torch.nn.Flatten(),
      torch.nn.Linear(3600, 120), # TODO: Understand this number
      torch.nn.Dropout(p=0.2),
      torch.nn.Tanh(), # Maybe usin Tanh
      torch.nn.Linear(120, last_size),
      torch.nn.Dropout(p=0.2),
    )
    return last_size

from dependencies.core import torch
from dependencies.datatypes import Parameter, Iterator
from .abs_module import AbsModule

class BasicConv(AbsModule):
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
    return 'BasicConv()'

  def __init__(self, **kwargs) -> None:
    """
      Constructor of the neural network.

      Parameters:
      -----------
        **kwargs: dict
          A dictionary containing the parameters
    """
    AbsModule.__init__(self, **kwargs)

    if 'features' not in kwargs and 'classifier' not in kwargs:
      dropout = kwargs.pop('dropout')
      last_size = self.__create_features__(dropout = dropout)
      self.__create_classifier__(last_size)

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
      torch.nn.LogSoftmax(dim=1),
    )

  def __create_features__(self, dropout: float = 0.0) -> int:
    """
      Creates the features.

      Returns:
      --------
        int
          The size of the last features, to pipe it to the classifier.
    """
    last_size = 96
    self.features = torch.nn.Sequential(
      torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(64),
      torch.nn.Dropout2d(dropout),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size=2, stride=2),

      torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(128),
      torch.nn.Dropout2d(dropout),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size=2, stride=2),

      torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(256, momentum=0.9),
      torch.nn.Dropout2d(dropout),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size=2, stride=2),

      torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(512, momentum=0.9),
      torch.nn.Dropout2d(dropout),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size=2, stride=2),

      torch.nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
      torch.nn.Flatten(),
      torch.nn.Linear(320*60, last_size),
    )
    return last_size

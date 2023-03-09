from dependencies.core import torch
from .abs_module import AbsModule

class BasicConv(AbsModule, torch.nn.Module):
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

  def __init__(self, init_features:bool = False, **kwargs) -> None:
    """
      Constructor of the neural network.

      Parameters:
      -----------
        init_features: bool
          A flag indicating if the features should be initialized with the module
        **kwargs: dict
          A dictionary containing the parameters
    """
    if init_features:
      dropout = kwargs.get('dropout', 0.0)
      momentum = kwargs.get('momentum', 0.1)
      labels = kwargs.get('labels', [])
      input_size = kwargs.get('input_size', 64)
      num_classes = kwargs.get('num_classes', len(labels))
      torch.nn.Module.__init__(self)
      self.__create_features__(dropout = dropout, momentum = momentum)
      self.__create_classifier__(input_size, num_classes)
      kwargs['features'] = self.features
      kwargs['classifier'] = self.classifier
      kwargs['state_dict'] = {}
    AbsModule.__init__(self, **kwargs)

  def __create_classifier__(self, last_size: int, num_classes: int) -> None:
    """
      Creates the classifier.

      Parameters:
      -----------
        last_size: int
          The size of the last layer.
    """
    self.classifier = torch.nn.Sequential(
      torch.nn.Flatten(),
      torch.nn.Linear(last_size, num_classes),
      torch.nn.ReLU(),
      torch.nn.LogSoftmax(dim=1),
    )

  def __create_features__(self, dropout: float = 0.0, momentum: float = 0.1) -> None:
    """
      Creates the features.

      Returns:
      --------
        int
          The size of the last features, to pipe it to the classifier.
    """
    self.features = torch.nn.Sequential(
      torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(64),
      torch.nn.Dropout2d(dropout),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size=2, stride=2),

      torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(128),
      torch.nn.Dropout2d(dropout),
      torch.nn.Tanh(),
      torch.nn.MaxPool2d(kernel_size=2, stride=2),

      torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(256, momentum=momentum),
      torch.nn.Dropout2d(dropout),
      torch.nn.Tanh(),
      torch.nn.MaxPool2d(kernel_size=2, stride=2),

      torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(512, momentum=momentum),
      torch.nn.Dropout2d(dropout),
      torch.nn.Tanh(),
      torch.nn.MaxPool2d(kernel_size=2, stride=2),

      torch.nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(64),
      torch.nn.ReLU(),
    )

from dependencies.core import torch
from dependencies.datatypes import Parameter, Iterator
from .abs_module import AbsModule

class Basic(AbsModule):
  """
    A basic neural network module. With randomly selected features.

    Attributes:
    -----------
      (inherits from AbsModule)

    Methods:
    --------
      (inherits from AbsModule)
  """
  def __repr__(self):
    """
      A string representation of the neural network.

      Returns:
      --------
        : str
          A string representation 'Basic()'.
    """
    return 'Basic()'

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
      input_size = kwargs.pop('input_size')
      last_size = self.__create_features__(input_size, dropout = dropout)
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
      torch.nn.LogSoftmax(dim=1)
    )

  def __create_features__(self, input_size, dropout: float = 0.0) -> int:
    """
      Creates the features.

      Returns:
      --------
        int
          The size of the last features, to pipe it to the classifier.
    """
    last_size = 96
    self.features = torch.nn.Sequential(
      torch.nn.Linear(input_size, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 42),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout),
      torch.nn.Linear(42, 28),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout),
      torch.nn.Linear(28, 512),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout),
      torch.nn.Linear(512, 128),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout),
      torch.nn.Linear(128, 64),
      torch.nn.ReLU(),
      torch.nn.Linear(64, last_size),
      torch.nn.LogSoftmax(dim=1)
    )
    return last_size

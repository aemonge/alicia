from dependencies.core import torch
from dependencies.datatypes import Parameter, Iterator
from .abs_module import AbsModule

class Basic(AbsModule):
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
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    """
      A forward pass of the neural network.

      Parameters:
      -----------
        x: torch.Tensor
          A batch of input features.

      Returns:
      --------
        torch.Tensor
    """
    return self.forward(x)

  def __repr__(self):
    """
      A string representation of the neural network.

      Returns:
      --------
        : str
          A string representation 'Basic()'.
    """
    return 'Basic()'

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
    last_size = 10
    if self.dropout > 0.0:
      self.features = torch.nn.Sequential(
        torch.nn.Linear(self.input_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 42),
        torch.nn.ReLU(),
        torch.nn.Dropout(self.dropout),
        torch.nn.Linear(42, 28),
        torch.nn.ReLU(),
        torch.nn.Dropout(self.dropout),
        torch.nn.Linear(28, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(self.dropout),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(self.dropout),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, last_size),
        torch.nn.LogSoftmax(dim=1)
      )
    else:
      self.features = torch.nn.Sequential(
        torch.nn.Linear(self.input_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 42),
        torch.nn.ReLU(),
        torch.nn.Linear(42, 28),
        torch.nn.ReLU(),
        torch.nn.Linear(28, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, last_size),
        torch.nn.LogSoftmax(dim=1)
      )
    return last_size

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
      A forward pass of the neural network.

      Parameters:
      -----------
        x: torch.Tensor
          A batch of input features.

      Returns:
      --------
        torch.Tensor

      Help:
      -----
        model.forward = lambda x: model.classifier(model.features(x)).view(x.size(0), class_count)
    """
    x = self.features(x)
    x = self.classifier(x)
    return torch.flatten(x, 1)

  def parameters(self) -> Iterator[Parameter]:
    """
      Get the parameters of the neural network.

      Returns:
      --------
        Iterator[Parameter]
    """
    return self.features.parameters()

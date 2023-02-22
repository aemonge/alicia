from dependencies.core import torch
from dependencies.datatypes import Parameter, Iterator
from modules.models.abs_module import AbsModule

class Elemental(AbsModule):
  """
    A elemental neural network module, only for testing purposes.

    Attributes:
    -----------
      labels: list
        A list of labels.
      features: list
        A list of features.
      input_size: int
        The input size.
      num_classes: int
        The number of classes.

    Methods:
    --------
      forward(self, x: torch.Tensor) -> torch.Tensor
        A forward pass of the neural network.
      load(self, path: str) -> None
        A load of the neural network.
      save(self, path: str) -> str
        A save of the neural network.
      create(self, input_size: int, num_classes: int) -> None
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
    return 'Elemental()'

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
      self.features = torch.nn.Sequential(
        torch.nn.Linear(self.input_size, self.num_classes),
        torch.nn.LogSoftmax(dim=1)
      )
    else:
      AbsModule.__init__(self, data = data)

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
    return torch.flatten(x, 1)

  def parameters(self) -> Iterator[Parameter]:
    """
      Get the parameters of the neural network.

      Returns:
      --------
        Iterator[Parameter]
    """
    return self.features.parameters()

  def create(self, *, input_size: int = 28, dropout: float|None = None) -> None: # pyright: ignore
    """
      Re creates the neural network.

      Parameters:
      -----------
        input_size: int
          The input size.
    """

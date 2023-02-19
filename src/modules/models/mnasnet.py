from dependencies.core import torch, torchvision
from dependencies.datatypes import Parameter, Iterator
from .abs_module import AbsModule

class Mnasnet(AbsModule, torchvision.models.MNASNet):
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
    return 'Squeezenet()'

  def parameters(self) -> Iterator[Parameter]:
    """
      Get the parameters of the neural network.

      Returns:
      --------
        Iterator[Parameter]
    """
    return self.features.parameters()

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
      super().__init__(alpha=1.0, num_classes=len(labels), dropout=dropout)
      self.input_size = input_size
      self.labels = labels
      self.training_history = []
      self.num_classes = len(labels)
      self.features = self.layers
    else:
      if 'dropout' in data:
        self.dropout = data['dropout']
      self.labels = data['labels']
      super().__init__(alpha=1.0, num_classes=len(self.labels), dropout=dropout)

      self.input_size = data['input_size']
      self.features = self.layers
      self.features = data['features']
      if 'training_history' in data:
        self.training_history = data['training_history']
      if 'dropout' in data:
        self.dropout = data['dropout']
      if 'classifier' in data:
        self.classifier = data['classifier']

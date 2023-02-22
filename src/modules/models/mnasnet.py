from .abs_module import AbsModule
from torchvision.models import MNASNet

class Mnasnet(MNASNet, AbsModule):
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
    return 'Mnasnet()'

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
    if num_classes is None:
      num_classes = len(labels)

    if data is None:
      if dropout > 0.0:
        AbsModule.__init__(self,
          labels = labels, input_size = input_size, dropout = dropout, num_classes = num_classes
        )
        MNASNet.__init__(self, alpha=1.0, num_classes = num_classes, dropout=dropout)
      else:
        AbsModule.__init__(self, labels = labels, input_size = input_size, num_classes = num_classes)
        MNASNet.__init__(self, alpha=1.0, num_classes = num_classes)
      self.features = self.layers
      # MNASNet destroys the labels attribute.
      self.labels = labels # pyright: reportGeneralTypeIssues=false
    else:
      AbsModule.__init__(self, data = data)
      self.layers = self.features

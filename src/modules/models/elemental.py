from dependencies.core import torch
from modules.models.abs_module import AbsModule

class Elemental(AbsModule, torch.nn.Module):
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
      input_size = kwargs.get('input_size', 28)
      labels = kwargs.get('labels', [])
      num_classes = kwargs.get('num_classes', len(labels))
      torch.nn.Module.__init__(self)
      self.features = torch.nn.Sequential(
        torch.nn.Linear(input_size, 64, bias=False),
        torch.nn.ReLU(),
      )
      self.classifier = torch.nn.Sequential(
        torch.nn.Linear(64, num_classes, bias=False),
        torch.nn.ReLU(),
        torch.nn.LogSoftmax(dim=1)
      )
      kwargs['features'] = self.features
      kwargs['classifier'] = self.classifier
      kwargs['state_dict'] = {}
    AbsModule.__init__(self, **kwargs)

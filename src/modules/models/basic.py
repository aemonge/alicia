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

  def __str__(self):
    """
      A verbose string representation of the neural network.

      Returns:
      --------
        : str
          labels, features, classifier
    """
    features_str = "\n  ".join(str(self.features).split("\n"))
    classifier_str = "\n  ".join(str(self.classifier).split("\n"))

    return "Basic(\n" + \
      f"  (labels): \n    {self.labels}\n" + \
      f"  (features): {features_str}\n" + \
      f"  (classifier): {classifier_str}\n" + \
    f")"

  def __init__(self, labels) -> None:
    """
      Constructor of the neural network.

      Parameters:
      -----------
        labels: list
          A list of labels.
    """
    super().__init__()
    self.labels = labels
    self.num_classes = len(labels)

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

  def __create_features__(self, dropout: float) -> int:
    """
      Creates the features.

      Parameters:
      -----------
        dropout: float
          The dropout probability.

      Returns:
      --------
        int
          The size of the last features, to pipe it to the classifier.
    """
    last_size = 10
    self.features = torch.nn.Sequential(
      torch.nn.Linear(self.input_size, 128),
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

  def load(self, path: str) -> None:
    """
      Parameters:
      -----------
        path: str
          path to load model

      Returns:
      --------
        None
    """
    data = torch.load(path)
    self.labels = data['labels']
    self.features = data['features']
    self.classifier = data['classifier']
    self.load_state_dict(data['state_dict'])

  def parameters(self) -> Iterator[Parameter]:
    """
      Get the parameters of the neural network.

      Returns:
      --------
        Iterator[Parameter]
    """
    return self.features.parameters()

  def save(self, path: str) -> None:
    """
      Save the neural network.

      Parameters:
      -----------
        path: str
          path to save model

      Returns:
      --------
        None
    """
    torch.save({
      'name': 'Basic',
      'labels': self.labels,
      'features': self.features,
      'classifier': self.classifier,
      'state_dict': self.state_dict(),
    }, path)

  def create(self, input_size: int = 28, dropout: float = 0.5) -> None:
    """
      Re creates the neural network.

      Parameters:
      -----------
        input_size: int
          The input size.
        dropout: float
          The dropout probability.
    """
    self.input_size = input_size
    last_size = self.__create_features__(dropout)
    self.__create_classifier__(last_size)

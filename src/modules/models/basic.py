from dependencies.core import torch
from dependencies.datatypes import Parameter, Iterator
from .abs_module import AbsModule

class Basic(AbsModule):
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    return self.forward(x)

  def __repr__(self):
    return 'Basic()'

  def __str__(self):
    features_str = "\n  ".join(str(self.features).split("\n"))
    classifier_str = "\n  ".join(str(self.classifier).split("\n"))

    return "Basic(\n" + \
      f"  (labels): \n    {self.labels}\n" + \
      f"  (features): {features_str}\n" + \
      f"  (classifier): {classifier_str}\n" + \
    f")"

  def __init__(self, labels) -> None:
    super().__init__()
    self.labels = labels
    self.num_classes = len(labels)

  def __create_classifier__(self, last_size: int) -> None:
    self.classifier = torch.nn.Sequential(
      torch.nn.Linear(last_size, self.num_classes),
      torch.nn.LogSoftmax(dim=1)
    )

  def __create_features__(self, dropout: float) -> int:
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
      help
      ---------
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
    return self.features.parameters()

  def save(self, path: str) -> None:
    """
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
    self.input_size = input_size
    last_size = self.__create_features__(dropout)
    self.__create_classifier__(last_size)

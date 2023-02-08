import random
import torch
import torch.nn as nn

from modules.models.AbsModule import AbsModule

class Basic(AbsModule):
  RANDOM_CHANNELS = lambda _ : random.sample([7, 42, 32, 64, 128, 101, 256], 1)[0]
  KERNEL_SIZE = (3, 3)
  STRIDE = (1, 1)
  PADDING = (1, 1)

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

  def __init__(self, labels: list) -> None:
    super().__init__()
    self.labels = labels
    self.num_classes = len(labels)

  def __create_classifier__(self, last_size: int) -> None:
    self.classifier = nn.Sequential(
      nn.Linear(last_size, self.num_classes),
      nn.LogSoftmax(dim=1)
    )

  def __create_features__(self, hidden_units: int, dropout: float) -> int:
    # TODO: Implement a random selection of nn.Functions such as Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d
    out_channels = self.RANDOM_CHANNELS()
    self.features = nn.Sequential(
      nn.Conv2d(self.input_size, out_channels, self.KERNEL_SIZE, self.STRIDE, self.PADDING),
      nn.ReLU()
    )

    for _ in range(hidden_units):
      in_channels = out_channels
      out_channels = self.RANDOM_CHANNELS()
      self.features.append(nn.Conv2d(in_channels, out_channels, self.KERNEL_SIZE, self.STRIDE, self.PADDING))
      self.features.append(nn.ReLU())
      self.features.append(nn.Dropout(p=dropout))

    # TODO: Remove code bellowReset to a hard-coded features to test
    last_size = 10
    self.features = torch.nn.Sequential(
      torch.nn.Linear(self.input_size, 784),
      torch.nn.ReLU(),
      torch.nn.Linear(784, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 64),
      torch.nn.ReLU(),
      torch.nn.Linear(64, last_size),
      torch.nn.LogSoftmax(dim=1)
    )
    return last_size

    return out_channels

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
    data = torch.load(path)
    self.labels = data['labels']
    self.features = data['features']
    self.classifier = data['classifier']
    self.load_state_dict(data['state_dict'])

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    return self.forward(x)

  # TODO: Do you really need to re-implement this? I would guess not bro
  # def train(self) -> None:
  #   self.features.train();
  #
  # def eval(self) -> None:
  #   self.features.eval();

  def parameters(self) -> list:
    return self.features.parameters()

  def save(self, path: str) -> None:
    torch.save({
      'name': 'Basic',
      'labels': self.labels,
      'features': self.features,
      'classifier': self.classifier,
      'state_dict': self.state_dict(),
    }, path)

  def create(self, input_size: int = 784, hidden_units: int = 6, dropout: float = 0.5) -> None:
    if hidden_units < 1 or hidden_units > 256:
      raise ValueError('hidden_units must be between 1 and 256')

    self.input_size = input_size
    last_size = self.__create_features__(hidden_units, dropout)
    self.__create_classifier__(last_size)

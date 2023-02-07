# Global deps
import torch
import time

# Local deps
from lib.dispaly_analytics import print_step, print_total
from datasets.alicia_dataset import AliciaDataset

# Types
from models.abstract import Module
from torch.utils.data import DataLoader

# Fancy
from loading_display import spinner
from loading_display import loading_bar
from termcolor import colored

class Trainer(object):
  BAR_LENGTH = 55
  LOADING_ICONS = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']

  def __init__(self, model: Module, transforms, learning_rate: float = 1/137,  momentum: float = 0.85) -> None:
    self.model = model
    self.criterion = torch.nn.CrossEntropyLoss()
    self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    self.transforms = transforms

  def train_step(self, tr_loss: float, images: torch.Tensor, labels: torch.Tensor) -> float:
    s = spinner(icons=[colored(i, 'blue') for i in self.LOADING_ICONS])
    next(s)

    self.optimizer.zero_grad()
    next(s)

    output = self.model.forward(images)
    next(s)

    loss = self.criterion(output, labels)
    next(s)

    loss.backward()
    next(s)

    self.optimizer.step()
    next(s)

    tr_loss += loss.item()
    next(s)

    return tr_loss

  def validation_step(self, dataloaders: DataLoader, batch_size: int):
    s = spinner(icons=[colored(i, 'green') for i in self.LOADING_ICONS])
    ix = 0
    vd_loss = 0.0
    vd_correct = 0
    validate_loader_count = len(dataloaders.dataset)

    with torch.no_grad():
      self.model.eval()
      next(s)
      for (images, (labels, _)) in iter(dataloaders):
        ix += batch_size * 1
        next(s)

        output = self.model.forward(images)
        next(s)

        loss = self.criterion(output, labels)
        next(s)

        vd_loss += loss.item()
        next(s)

        ps = torch.exp(output)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        vd_correct += equals.sum().item()
        next(s)

        loading_bar(
          ix, total=validate_loader_count,
          bar_length=self.BAR_LENGTH, show_percentage=True, icon=colored('⠿', 'green')
        )

    next(s)
    self.model.train()
    return vd_loss, vd_correct

  def train(self, data_dir: str, batch_size: int = 64, epochs: int = 1, freeze_parameters: bool = False) -> None:
    if self.transforms is None or self.transforms['train'] is None  or self.transforms['valid'] is None:
      raise ValueError('Transforms must be defined and set')

    train_ldr = DataLoader(
        AliciaDataset(f"{data_dir}/train", transform = self.transforms['train']),
        batch_size = batch_size, shuffle=True
    )
    valid_ldr = DataLoader(
        AliciaDataset(f"{data_dir}/valid", transform = self.transforms['valid']),
        batch_size = batch_size, shuffle=True
    )

    train_loader_count = len(train_ldr.dataset)
    validate_loader_count = len(valid_ldr.dataset)

    if freeze_parameters:
      for param in self.model.features.parameters():
        param.requires_grad = False


    print(f" Epochs: {epochs}, Items: [training: \"{train_loader_count:,}\" , validation: \"{validate_loader_count:,}\"]")

    time_count = 0
    vd_correct = 0
    total_time = time.time()
    start_time = total_time

    for epoch in range(epochs):
      tr_loss = 0.0
      ix = 0

      print(f"   Epoch: {epoch + 1}/{epochs} ({colored('traning', 'blue')} and {colored('validating', 'green')})")
      for (images, (labels, _)) in iter(train_ldr):
        ix += batch_size * 1
        tr_loss = self.train_step(tr_loss, images, labels)
        loading_bar(
          ix, total=train_loader_count,
          bar_length=self.BAR_LENGTH, show_percentage=True, icon=colored('⠿', 'blue')
        )
      else:
        print()
        vd_loss, vd_correct = self.validation_step(valid_ldr, batch_size)

        print('\033[F\033[K', end='\r') # back prev line and clear
        print('\033[F\033[K', end='') # back prev line and clear
        time_now = print_step(
          epoch, epochs, start_time, time_count,
          tr_loss, vd_loss, vd_correct,
          validate_loader_count, train_loader_count
        )
        start_time = time_now
    else:
      print()
      print_total(vd_correct, validate_loader_count, total_time)

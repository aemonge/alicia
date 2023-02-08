# Global deps
import torch
import time
from PIL import Image
from matplotlib import pyplot as plt
from random import randrange
from wcmatch import glob
import os

# Local deps
from views.shared.DispalyAnalytics import print_t_step, print_step, print_total
from views.shared.PyplotHelper import imshow, print_pbs
from modules.datasets.UnLabeledImageDataset import UnLabeledImageDataset

# Types
from modules.models.AbsModule import AbsModule
from torch.utils.data import DataLoader

# Fancy
from loading_display import spinner
from loading_display import loading_bar
from termcolor import colored

class Trainer:
  BAR_LENGTH = 55
  LOADING_ICONS = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']

  def __init__(self, model: AbsModule, transforms, learning_rate: float = 1/137,  momentum: float = 0.85) -> None:
    self.model = model
    self.learning_rate = learning_rate
    self.momentum = momentum

    self.criterion = torch.nn.CrossEntropyLoss()
    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
    self.transforms = transforms

  def train_step(self, tr_loss: float, images: torch.Tensor, labels: torch.Tensor) -> float:
    s = spinner(icons=[' ' + colored(i, 'blue') for i in self.LOADING_ICONS])
    next(s)

    self.optimizer.zero_grad()
    next(s)

    # TODO: use model()
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
    s = spinner(icons=[' ' + colored(i, 'green') for i in self.LOADING_ICONS])
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

  def train(self, data_dir: str, labels: dict,
            batch_size: int = 64, epochs: int = 1, freeze_parameters: bool = True) -> None:
    """
      help:
      ----------
        print(labels['25.jpg']) # -> Str
        print(category_labels_ids[labels['25.jpg']]) # -> Int
    """
    if self.transforms is None or self.transforms['train'] is None  or self.transforms['valid'] is None:
      raise ValueError('Transforms must be defined and set')

    category_labels_ids = { v:k for k,v in enumerate(self.model.labels)}
    train_ldr = DataLoader(UnLabeledImageDataset(
        f"{data_dir}/train", labels, category_labels_ids, transform = self.transforms['train']
      ), batch_size = batch_size, shuffle=True
    )
    valid_ldr = DataLoader(UnLabeledImageDataset(
        f"{data_dir}/valid", labels, category_labels_ids, transform = self.transforms['valid']
      ), batch_size = batch_size, shuffle=True
    )

    train_loader_count = len(train_ldr.dataset)
    validate_loader_count = len(valid_ldr.dataset)

    if freeze_parameters:
      # TODO: Use self.model
      for param in self.model.parameters():
        param.requires_grad = False


    print(f" Epochs: {epochs}, Learning rate: {self.learning_rate}, Momentum: {self.momentum},",
          f" Items: [training: \"{train_loader_count:,}\" , validation: \"{validate_loader_count:,}\"]\n")

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
        vd_loss, vd_correct = self.validation_step(valid_ldr, batch_size)

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

  def test(self, data_dir: str, labels: dict, batch_size: int = 64):
    if self.transforms is None or self.transforms['test'] is None:
      raise ValueError('Test or valid transforms must be defined and set')

    s = spinner(icons=[colored(i, 'yellow') for i in self.LOADING_ICONS])
    t_correct = 0
    ix = 0
    start_time = time.time()
    category_labels_ids = { v:k for k,v in enumerate(self.model.labels)}
    test_ldr = DataLoader(UnLabeledImageDataset(
      f"{data_dir}/test", labels, category_labels_ids, transform = self.transforms['valid']
      ), batch_size = batch_size, shuffle=True
    )
    test_loader_count = len(test_ldr.dataset)

    with torch.no_grad():
      self.model.eval()
      next(s)
      for (images, (labels, _)) in iter(test_ldr):
        ix += batch_size * 1
        next(s)

        output = self.model(images)
        next(s)

        ps = torch.exp(output)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        t_correct += equals.sum().item()
        next(s)

        loading_bar(
          ix, total=test_loader_count,
          bar_length=self.BAR_LENGTH, show_percentage=True, icon=colored('⠿', 'yellow')
        )
      else:
        print('\033[F\033[K', end='') # back prev line and clear
        print_t_step(start_time, t_correct, test_loader_count)

  def show_test_results(self, path: str, labels: dict, count: int, tilted: bool = False):
    all_imgs = [
      file for file in glob.glob(f"{path}/**/*.@(jpg|jpeg|png)", flags=glob.EXTGLOB)
    ]

    fig, ax = plt.subplots(count, 2, width_ratios=[4,1])
    if tilted:
      plt.subplots_adjust(left=0.22, bottom=0.08, top=0.98, hspace=0.6, wspace=0)
    else:
      plt.subplots_adjust(left=0.22, bottom=0.08, top=0.90, hspace=0.6, wspace=0.35, right=0.86)

    for c_idx in range(count):
      ix = randrange(len(all_imgs))
      img_path = all_imgs[ix]

      real_class = [
        'Ankle boot', 'Bag', 'Coat', 'Dress', 'Pullover', 'Sandal', 'Shirt',
        'Sneaker', 'Top', 'Trouser'
      ]
      image = Image.open(img_path)
      file_name = os.path.basename(img_path)

      # tensor_img = self.transforms['test'](image)
      probs, idxs = self.predict(image, topk=3)
      class_labels = [ self.model.labels[i] for i in idxs]

      print_pbs(class_labels, probs, ax=ax[c_idx][0])
      imshow(self.transforms['display'](image), title=labels[file_name], ax=ax[c_idx][1], tilted=tilted)

    fig.align_ylabels(ax[:, 1])
    plt.show()

  def predict(self, image, topk=5):
    with torch.no_grad():
      self.model.eval()
      tensor_img = self.transforms['test'](image)
      # TODO: use self.model()
      logps = self.model(tensor_img.unsqueeze(0)) # from 3d to 4d [ introducing a batch dimension ]
      ps = logps[0] # Return to 3D [ no batches again ]
      ps_val, ps_idx = ps.topk(topk)

      return ps_val.numpy(), ps_idx.numpy()

# Global deps
import math
import torch
import time
from PIL import Image
from matplotlib import pyplot as plt
from random import randrange
from wcmatch import glob
import os

# Local deps
from libs.Pretty_Info import PrettyInfo
from modules.datasets.UnLabeledImageDataset import UnLabeledImageDataset

# Types
from modules.models.AbsModule import AbsModule
from torch.utils.data import DataLoader

class Trainer(PrettyInfo):
  def __init__(self, model: AbsModule, transforms, learning_rate: float = 1/137,  momentum: float|None = None) -> None:
    super().__init__()
    self.model = model
    self.learning_rate = learning_rate

    self.criterion = torch.nn.CrossEntropyLoss()
    if momentum is not None:
      self.momentum = momentum
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
    else:
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
    self.transforms = transforms

  def train_step(self, tr_loss: float, images: torch.Tensor, labels: torch.Tensor) -> float:
    self._spin()

    self.optimizer.zero_grad()
    self._spin()

    output = self.model(images)
    self._spin()

    loss = self.criterion(output, labels)
    self._spin()

    loss.backward()
    self._spin()

    self.optimizer.step()
    self._spin()

    tr_loss += loss.item()
    self._spin()

    return tr_loss

  def validation_step(self, dataloaders: DataLoader, batch_size: int) -> tuple[float, int]:
    ix = 0
    vd_loss = 0.0
    vd_correct = 0
    validate_loader_count = len(dataloaders.dataset)

    with torch.no_grad():
      self.model.eval()
      self._spin(step='valid')

      for (images, (labels, _)) in iter(dataloaders):
        ix += batch_size * 1
        self._spin(step='valid')

        output = self.model.forward(images)
        self._spin(step='valid')

        loss = self.criterion(output, labels)
        self._spin(step='valid')

        vd_loss += loss.item()
        self._spin(step='valid')

        ps = torch.exp(output)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        vd_correct += equals.sum().item()
        self._spin(step='valid')

        self._loading(ix, validate_loader_count, step = 'valid')

    self._spin(step='valid')
    self.model.train()
    return vd_loss, vd_correct

  def train(self, data_dir: str, labels: dict, batch_size: int = 64, epochs: int = 1,
            freeze_parameters: bool = False) -> None:
    """
      help:
      ----------
        labels['25.jpg'] # -> Str
        category_labels_ids[labels['25.jpg']] # -> Int
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

    self._print_train_header(epochs, batch_size, train_loader_count, validate_loader_count)

    time_count = 0
    vd_correct = 0
    total_time = time.time()
    start_time = total_time

    # Freeze parameters
    if (freeze_parameters):
      for param in self.model.features.parameters():
          param.requires_grad = False

    for epoch in range(epochs):
      tr_loss = 0.0
      ix = 0

      self._print_step_header(epochs, epoch)
      for (images, (labels, _)) in iter(train_ldr):
        ix += batch_size * 1
        tr_loss = self.train_step(tr_loss, images, labels)
        self._loading(ix, train_loader_count)
        if math.isnan(tr_loss):
          raise Exception('Loss has been lost, check parameters')

      else:
        vd_loss, vd_correct = self.validation_step(valid_ldr, batch_size)

        self.__backspace__(hard=True)
        time_now = self._print_step(
          epoch, epochs, start_time, time_count,
          tr_loss, vd_loss, vd_correct,
          validate_loader_count, train_loader_count
        )
        start_time = time_now
    else:
      self._print_total(vd_correct, validate_loader_count, total_time)

  def test(self, data_dir: str, labels: dict, batch_size: int = 64, freeze_parameters: bool = False):
    if self.transforms is None or self.transforms['test'] is None:
      raise ValueError('Test or valid transforms must be defined and set')

    t_correct = 0
    ix = 0
    start_time = time.time()
    category_labels_ids = { v:k for k,v in enumerate(self.model.labels)}
    test_ldr = DataLoader(UnLabeledImageDataset(
      f"{data_dir}/test", labels, category_labels_ids, transform = self.transforms['valid']
      ), batch_size = batch_size, shuffle=True
    )
    test_loader_count = len(test_ldr.dataset)
    self._print_test_header(batch_size, test_loader_count)

    if freeze_parameters: # TODO: I think isn't needed with torch.no_grad(), investigate
      for param in self.model.parameters():
        param.requires_grad = False

    with torch.no_grad():
      self.model.eval()
      self._spin(step='test')

      for (images, (labels, _)) in iter(test_ldr):
        ix += batch_size * 1
        self._spin(step='test')

        output = self.model(images)
        self._spin(step='test')

        ps = torch.exp(output)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        t_correct += equals.sum().item()
        self._spin(step='test')

        self._loading( ix, test_loader_count, step = 'test')
      else:
        self._print_t_step(start_time, t_correct, test_loader_count)

  def show_test_results(self, path: str, labels: dict, count: int, tilted: bool = False):
    all_imgs = [
      file for file in glob.glob(f"{path}/**/*.@(jpg|jpeg|png)", flags=glob.EXTGLOB)
    ]

    _, ax = plt.subplots(count, 2, width_ratios=[4,1])
    if count == 1:
      ax = [ax]

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

      self._print_pbs(class_labels, probs, ax=ax[c_idx][0])
      self._imshow(self.transforms['display'](image), title=labels[file_name], ax=ax[c_idx][1], tilted=tilted)

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

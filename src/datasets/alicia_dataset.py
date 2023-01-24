# pylint: disable=missing-module-docstring
import os as Os
import re as Re
import csv as Csv
import pathlib as Pathlib
import torch as Torch

from torchvision import transforms as Transforms
from natsort import natsorted as Natsorted
from PIL import Image
from torch.utils.data import Dataset
# pylint: enable=missing-module-docstring

BATCH_SIZE = 4

class AliciaDataset(Dataset):
  """
    Custom dataset for loading data from a folder, and labels form the folder file `labels.csv`

    Attributes
    ----------
      main_dir : str
        Path to the folder containing the images
      class_map : dict
        The Clases fetch from the labels files, which the images will be tested to match {'cat': 0, 'dog': 1}
      transform : callable
        Transform to be applied to the images
      label_transform : callable
        Transform to be applied to the labels

    Methods
    -------
  """

  LABELS_FILENAME = 'labels.csv'

  def __init__(self, main_dir, class_map = None, transform = None, label_transform = None):
    """

      Parameters
      ----------
        main_dir : str
          Path to the folder containing the images.
        class_map : dict
          The Clases fetch from the labels files, which the images will be tested to match.
        transform : callable, optional
          Optional transform to be applied on a sample.
        label_transform : callable, optional
          Optional transform to be applied on the labels.
    """
    all_imgs = _ = [f for f in Os.listdir(main_dir) if Re.match(r'[0-9]+.*\.jpg', f)]
    self.main_dir = main_dir
    self.transform = transform
    self.label_transform = label_transform
    labels_file = Pathlib.Path(self.main_dir, self.LABELS_FILENAME)

    self.__total_imgs = Natsorted(all_imgs)
    self.__labels = {}

    if class_map is None:
      class_map = set()

    for img in all_imgs:
      self.__labels[Pathlib.Path(img).stem] = None

    if labels_file.is_file():
      with labels_file.open(encoding="utf-8") as file:
        reader = Csv.reader(file)
        for file_name, label in reader:
          if isinstance(class_map, set):
            class_map.add(label)
          self.__labels[file_name] = label

    if isinstance(class_map, set):
      self.class_map = { x:i for i,x in enumerate(list(class_map)) }
    elif isinstance(class_map, dict):
      self.class_map = class_map

  def __len__(self):
    """
      Returns the number of samples in the dataset.

      Parameters
      ----------
        None

      Returns
      -------
        int
          The number of samples in the dataset.
    """
    return len(self.__total_imgs)

  def __getitem__(self, idx):
    """
      Returns a sample from the dataset.

      Parameters
      ----------
        idx : integer
          Index of the image to be loaded.

      Returns
      -------
        image,[class_tensor, file_name] : tuple(torch.Tensor, list(torch.Tensor, string))
          Batch image, class label (if any, empty tensor when no class) and its file name.
    """
    file = Pathlib.Path(self.main_dir, self.__total_imgs[idx])
    image = Image.open(file.as_posix()).convert("RGB")
    class_id = self.__labels[file.name]

    if self.transform:
      image = self.transform(image)
    else:
      image = Transforms.Compose([Transforms.ToTensor()])(image)

    if self.label_transform:
      print('Not implemented')

    if class_id is not None:
      class_id = self.class_map[class_id]
      class_tensor = Torch.tensor(class_id)

      return image, [class_tensor, file.name]
    # else
    return image, [Torch.tensor([]), file.name]

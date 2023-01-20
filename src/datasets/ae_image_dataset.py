# pylint: disable=missing-module-docstring
import os as Os
import re as Re
import csv as Csv
import torch as Torch

from torchvision import transforms as Transforms
from natsort import natsorted as Natsorted
from PIL import Image
from torch.utils.data import Dataset
# pylint: enable=missing-module-docstring

BATCH_SIZE = 4

class AeImageDataset(Dataset):
  """
    Custom dataset for loading data from a folder, and labels form the folder file `labels.csv`

    Attributes
    ----------
      main_dir : str
        Path to the folder containing the images
      class_map : dict
        The Clases fetch from the labels files, which the images will be tested to match
      transform : callable
        Transform to be applied to the images
      label_transform : callable
        Transform to be applied to the labels

    Methods
    -------
  """

  def __init__(self, main_dir, class_map = set(), transform = None, label_transform = None):
    """

      Parameters
      ----------
        main_dir : str
          Path to the folder containing the images
        transform : callable, optional
          Optional transform to be applied on a sample.
        label_transform : callable, optional
          Optional transform to be applied on the labels.
    """
    all_imgs = files = [f for f in Os.listdir(main_dir) if Re.match(r'[0-9]+.*\.jpg', f)]
    self.main_dir = main_dir
    self.transform = transform
    self.label_transform = label_transform

    self.__total_imgs = Natsorted(all_imgs)
    self.__labels = dict()

    with open(f"{self.main_dir}/labels.csv") as file:
      reader = Csv.reader(file)
      for row in reader:
        class_map.add(row[1])
        self.__labels[row[0]] = row[1]

    self.class_map = { x:i for i,x in enumerate(list(class_map)) }

  def __len__(self):
    """
      Returns the number of samples in the dataset
    """
    return len(self.__total_imgs)

  def __getitem__(self, idx):
    """
      Returns a sample from the dataset

      Parameters
      ----------
      idx : integer
        Index of the image to be loaded.

      Returns
      -------
      image,class_id : tuple(torch.Tensor, torch.Tensor)
        Batch image and its class label.
    """
    img_loc = Os.path.join(self.main_dir, self.__total_imgs[idx])
    img_rg = Re.search(r'\/(.[^\/]*)\.(jpg|png)', img_loc)
    file_name = f"{img_rg[1]}.{img_rg[2]}"
    img_name = img_rg[1]
    image = Image.open(img_loc).convert("RGB")
    class_id = self.__labels[img_name]

    if self.transform:
      image = self.transform(image)
    else:
      image = Transforms.Compose([Transforms.ToTensor()])(image)

    if self.label_transform:
      print('Not implemented')

    class_id = self.class_map[class_id]
    class_tensor = Torch.tensor(class_id)

    return image, [class_tensor, file_name]

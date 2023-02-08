# pylint: disable=missing-module-docstring
from wcmatch import glob
import pathlib
import torch

from torchvision import transforms as Transforms
from PIL import Image
from torch.utils.data import Dataset
# pylint: enable=missing-module-docstring

BATCH_SIZE = 4

class UnLabeledImageDataset(Dataset):
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

  def __init__(self, main_dir, labels: dict|None = None, labels_ids: dict|None = None, transform = None):
    """

      Parameters
      ----------
    """
    self.__imgs = [file for file in glob.glob(f"{main_dir}**/*.@(jpg|jpeg|png)", flags=glob.EXTGLOB)]
    self.main_dir = main_dir
    self.transform = transform
    self.__labels = labels
    self.__labels_ids = labels_ids

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
    return len(self.__imgs)

  def __getitem__(self, idx) -> tuple[Image.Image|torch.Tensor, tuple[torch.Tensor, str]]:
    """
      Returns a sample from the dataset.

      Parameters
      ----------
        idx : integer
          Index of the image to be loaded.

      Returns
      -------
        image,[class_tensor, file_name] : tuple(torch.Tensor, tuple(torch.Tensor, str))
          Batch image, class label (if any, empty tensor when no class) and its file name.
    """
    file = pathlib.Path(self.__imgs[idx])
    image = Image.open(file.as_posix()).convert("RGB")

    if self.transform:
      image = self.transform(image)
    else:
      image = Transforms.Compose([Transforms.ToTensor()])(image)

    if self.__labels is None:
      return image, (torch.tensor([]), file.name)

    class_name = self.__labels[file.name]
    class_id = self.__labels_ids[class_name]

    return image, (torch.tensor(class_id), file.name)

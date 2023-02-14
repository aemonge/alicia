from dependencies.core import glob, pathlib, torch, torchvision, Image
from dependencies.datatypes import Dataset

BATCH_SIZE = 4

class UnLabeledImageDataset(Dataset):
  """
    Custom dataset for loading data from a folder, and labels form the folder file `labels.csv`

    Attributes
    ----------
      main_dir : str
        Path to the folder containing the images
      transform: torchvision.transforms.Compose
        Transform to be applied to the images

    Methods
    -------
  """

  def __init__( self, main_dir, labels: dict, labels_ids: dict, *, transform: torchvision.transforms.Compose):
    """
      Parameters
      ----------
        main_dir: str
          Path to the folder containing the images
        labels: dict
          The Clases fetch from the labels files, which the images will be tested to match {'cat': 0, 'dog': 1}
        labels_ids: dict
          The ids of the labels, which the images will be tested to match {'cat': 0, 'dog': 1}
        transform: torchvision.transforms.Compose
          Transform to be applied to the images

      Returns
      -------
        None
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

    image = self.transform(image)
    class_name = self.__labels[file.name]
    class_id = self.__labels_ids[class_name]

    return image, (torch.tensor(class_id), file.name)

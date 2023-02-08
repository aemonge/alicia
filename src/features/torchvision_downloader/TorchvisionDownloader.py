# pylint: disable=missing-module-docstring
# pylint: disable=multiple-imports
import glob, os, os.path, contextlib, io
import pathlib as Pathlib
from termcolor import colored
import torchvision
# pylint: enable=missing-module-docstring

class TorchvisionDownloader():
  """
    Downloads a dataset from torchvision.

    Attributes
    ----------
      dataset_name : str
        The name of the dataset to download.
      train_dir : str
        The directory to download the training data to.
      test_dir : str
        The directory to download the test data to.
      split_percentage : integer
        The percentage of the dataset to download.

    Methods
    -------
      call : None
        Downloads the dataset.

  """
  def __init__(self, dir = 'data', dataset = 'MNIST', split_percentage = 70):
    """
      Constructor

      Parameters
      ----------
        test_dir : str, optional
          Path to the directory to save the split of `./train` and `./test`. The default is './data'.
        dataset : str, optional
          Name of the dataset. The default is 'MNIST'.
        split_percentage : int, optional
          Percentage of the dataset to be used for training. The default is 70.

      Returns
      -------
        None
    """
    self.__dataset = None
    self.__tmp_path = "tmp"
    self.dataset_name = dataset
    self.split_percentage = split_percentage
    self.__train_labels = []
    self.__test_labels = []

    self.train_dir = Pathlib.Path(dir, 'train')
    self.test_dir = Pathlib.Path(dir, 'test')

    if not self.train_dir.exists():
      os.mkdir(self.train_dir)
    else:
      raise Exception("Train directory already exists!")

    if not self.test_dir.exists():
      os.mkdir(self.test_dir)
    else:
      raise Exception("Test directory already exists!")

  def call(self):
    """
      Downloads the dataset from torchvision.

      Parameters
      ----------
        None

      Returns
      -------
        None
    """
    print(colored(' Downloading ... 🌑 ', 'blue', attrs=['bold']), end='\r')
    self.__download_dataset()

    if self.dataset_name == 'FashionMNIST':
      self.__dataset = torchvision.datasets.FashionMNIST(self.__tmp_path)
    else:
      self.__dataset = torchvision.datasets.MNIST(self.__tmp_path)

    print(end='\x1b[2K')
    print(colored(' Processing ...  🌕 ', 'yellow', attrs=['bold']), end='\r')
    self.__iterate_dataset(self.__write_images)
    self.__write_labels()
    self.__clear_tmp()
    print(end='\x1b[2K')
    print(colored(' Ready           💚', 'green', attrs=['bold']))

  def __download_dataset(self):
    """
      Downloads the dataset from the internet and saves it in the tmp folder.

      Parameters
      ----------
        None

      Returns
      -------
        None
    """
    # TODO: Check if the data has been already downloaded with the dataset, not by file path
    if not os.path.exists(self.__tmp_path):
      os.mkdir(self.__tmp_path)

    with contextlib.redirect_stdout(io.StringIO()):
      with contextlib.redirect_stderr(io.StringIO()):
        if self.dataset_name == 'MNIST':
          self.__dataset = torchvision.datasets.MNIST(self.__tmp_path, download=True)
        elif self.dataset_name == 'FashionMNIST':
          self.__dataset = torchvision.datasets.FashionMNIST(self.__tmp_path, download=True)

  def __clear_tmp(self):
    """
      Clear the tmp directory

      Parameters
      ----------
        None

      Returns
      -------
        None
    """
    filelist = glob.glob(f"{self.__tmp_path}/{self.dataset_name}/raw/*")
    for file in filelist:
      os.remove(file)
    os.rmdir(f"{self.__tmp_path}/{self.dataset_name}/raw")
    os.rmdir(f"{self.__tmp_path}/{self.dataset_name}")

  def __idx_to_img(self, idx):
    """
      Convert the index to the image filename

      Parameters
      ----------
        idx : int
            The index of the image

      Returns
      -------
        str
          The filename for the image
    """
    return f"{idx}.jpg"

  def __iterate_dataset(self, cb_fn):
    """
      Iterates over the dataset and calls the callback function

      Parameters
      ----------
        cb_fn : function
          Callback function to be called on each image

      Returns
      -------
        None
    """
    idx = 0
    max_train_idx = int(len(self.__dataset) * self.split_percentage / 100)

    for img, label in self.__dataset:
      label = self.__custom_label_mapping(label)

      if idx < max_train_idx:
        cb_fn(img, idx, self.train_dir)
        self.__train_labels.append(f"{self.__idx_to_img(idx)},{label}")
      else:
        cb_fn(img, idx, self.test_dir)
        self.__test_labels.append(f"{self.__idx_to_img(idx)},{label}")

      idx+=1

  def __write_images(self, img, idx, path):
    """
      Write the image to the specified path transforming the .mat file into images.

      Parameters
      ----------
        img : torch.Tensor
          The image tensor
        idx : int
          The index of the dataset, that will be transformed into the filename of the image
        path : str
          The path where the image will be written

      Returns
      -------
        None
    """
    file = Pathlib.Path(path, self.__idx_to_img(idx))
    img.save(file.as_posix())

  def __custom_label_mapping(self, label):
    """
      Custom label mapping for the dataset, specifically map the FashionMNIST words to an index.

      Parameters
      ----------
        label : str
          The label to be mapped.

      Returns
      -------
        int
          The integer that represents the mapping.
    """
    if self.dataset_name == 'FashionMNIST':
      return [
        'Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
      ][label]
    # else:
    return label

  def __write_labels(self):
    """
      Write the labels to a CSV files from the test and train directories.

      Parameters
      -------
        None

      Returns
      -------
        None
    """
    train_file = Pathlib.Path(self.train_dir, 'labels.csv')
    test_file = Pathlib.Path(self.test_dir, 'labels.csv')

    with train_file.open(mode = 'w', encoding='utf-8') as file:
      for label in self.__train_labels:
        file.write(label)
        file.write('\n')
    file.close()

    with test_file.open(mode = 'w', encoding='utf-8') as file:
      for label in self.__test_labels:
        file.write(label)
        file.write('\n')
    file.close()

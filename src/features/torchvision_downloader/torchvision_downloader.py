from torchvision.datasets import Places365
from dependencies.core import glob, os, pathlib, torchvision, tempfile, inspect, json, pkg_resources
from dependencies.fancy import colored
from dependencies.datatypes import *
import configs

class TorchvisionDownloader:
  """
    Downloads a dataset from torchvision.

    Attributes
    ----------
      dataset_name : str
        The name of the dataset to download.
      dir : str
        The directory to download the training data to.
      split_percentage : tuple[float, float, float]
        The percentages of how to split the data set into three parts.

    Methods
    -------
      call : None
        Downloads the dataset.

  """
  def __init__(self, dir: str, dataset: str,
               split_percentage: tuple[float, float, float] = (0.65, 0.25, 0.1),
               **dataset_kwargs
               ):
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
    self.__dataset : list[Dataset[list[ImageDT]]]
    self.__tmp_path = tempfile.gettempdir()
    self.dataset_name = dataset
    self.dataset_kwargs = dataset_kwargs
    self.dataset_kwargs["root"] = self.__tmp_path
    self.dataset_kwargs["download"] = True
    self.split_percentage = split_percentage
    self.__labels = []

    self.__root_dir = dir
    self.__train_dir = pathlib.Path(dir, 'train')
    self.__valid_dir = pathlib.Path(dir, 'valid')
    self.__test_dir = pathlib.Path(dir, 'test')

    if not pathlib.Path(self.__root_dir).exists():
      os.mkdir(self.__root_dir)
      os.mkdir(self.__train_dir)
      os.mkdir(self.__valid_dir)
      os.mkdir(self.__test_dir)
    else:
      if self.__train_dir.exists() or self.__valid_dir.exists() or self.__test_dir.exists():
        self.__clear_tmp()
        raise Exception(f"{self.__root_dir} directory is not empty.")
      else:
        os.mkdir(self.__train_dir)
        os.mkdir(self.__valid_dir)
        os.mkdir(self.__test_dir)

  def __get_all_posible_splits(self) -> list[Dataset[list[ImageDT]]]:
    """
      Gets all the possible splits of the dataset.

      Returns
      -------
        list[Dataset[list[ImageDT]]]
    """
    datasets = []
    with pkg_resources.open_text(configs, 'dataset_split_methods.json') as f:
      specific_split = json.load(f)

    try:
      dataset = getattr(torchvision.datasets, self.dataset_name)
      if self.dataset_name in specific_split:
        for method, values in specific_split[self.dataset_name].items():
          for value in values:
            datasets.append(dataset(**{**self.dataset_kwargs, **{ method: value }}))
      else:
        datasets.append(dataset(**self.dataset_kwargs))
    except Exception as msg:
      raise Exception(f"Target dataset cannot be downloaded!, please try another one.\n\t\t{msg}")

    return datasets

  def call(self, *, categories: dict = None) -> None:
    """
      Downloads the dataset from torch-vision.

      Parameters
      ----------
        forced : bool, optional
          Force the download of the dataset. The default is False.

      Returns
      -------
        None
    """
    print(colored(' Downloading ... 🌑 ', 'blue', attrs=['bold']), end='\r')
    self.__dataset = self.__get_all_posible_splits()

    if categories:
      self._idx_to_class = categories
    else:
      try:
        items = [ i.class_to_idx.items() for i in self.__dataset]
        items = [item for sublist in items for item in sublist] # Flatten
        self._idx_to_class = {val:key for key, val in items}
      except:
        raise Exception("Target dataset does not have `class_to_idx`, provide a categories file")

    print('\r', end='\r')
    print(colored(' Processing ...  🌕 ', 'yellow', attrs=['bold']), end='\r')
    self.__iterate_dataset(self.__write_images)
    self.__write_labels()
    self.__clear_tmp()
    print('\r', end='\r')
    print(colored(' Ready           💚', 'green', attrs=['bold']))

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
    filelist = glob.glob(f"{self.__tmp_path}/{self.dataset_name}/**/*")
    for file in filelist:
      os.remove(file)
    try:
      os.rmdir(f"{self.__tmp_path}/{self.dataset_name}/raw")
      os.rmdir(f"{self.__tmp_path}/{self.dataset_name}")
    except:
      pass

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
    total =  sum([len(ds) for ds in self.__dataset]) # pyright: reportGeneralTypeIssues=false
    max_train_idx = int(total * self.split_percentage[0])
    max_valid_idx = int(total * (self.split_percentage[0] + self.split_percentage[1]))

    for ds in self.__dataset:
      for img, label in ds:
        label = self.__custom_label_mapping(label)

        if idx < max_train_idx:
          cb_fn(img, idx, self.__train_dir)
        elif idx < max_valid_idx:
          cb_fn(img, idx, self.__valid_dir)
        else:
          cb_fn(img, idx, self.__test_dir)

        self.__labels.append(f"{self.__idx_to_img(idx)},{label}")
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
    file = pathlib.Path(path, self.__idx_to_img(idx))
    img.save(file.as_posix())

  def __custom_label_mapping(self, label):
    """
      Custom label mapping for the dataset words to an index.

      Parameters
      ----------
        label : str
          The label to be mapped.

      Returns
      -------
        int
          The integer that represents the mapping.
    """
    if hasattr(self, '_idx_to_class'):
      try:
        return self._idx_to_class[label]
      except:
        return self._idx_to_class[str(label)]
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
    path = pathlib.Path(self.__root_dir, 'labels.csv')
    with path.open(mode = 'w', encoding='utf-8') as file:
      for label in self.__labels:
        file.write(label)
        file.write('\n')
    file.close()

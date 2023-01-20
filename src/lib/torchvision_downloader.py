import torchvision
import glob, os, os.path, contextlib, io
from termcolor import colored

class TorchvisionDownloader(object):
  def __init__(self, test_dir = 'data/test', train_dir = 'data/train', dataset = 'MNIST', split_percentage = 70):

    self.datasetName = dataset
    self.__dataset = None
    self.__tmp_path = f"tmp"
    self.train_dir = train_dir
    self.test_dir = test_dir
    self.split_percentage = split_percentage
    self.__train_labels = []
    self.__test_labels = []

  def call(self):
    if not os.path.exists(self.__tmp_path) or not os.path.exists(f"{self.__tmp_path}/{self.datasetName}"):
      print(colored(' Downloading ... 🌑 ', 'blue', attrs=['bold']), end='\r')
      self.__downloadDataset()
    elif self.datasetName == 'FashionMNIST':
      self.__dataset = torchvision.datasets.FashionMNIST(self.__tmp_path)
    else:
      self.__dataset = torchvision.datasets.MNIST(self.__tmp_path)

    print(end='\x1b[2K')
    print(colored(' Processing ...  🌕 ', 'yellow', attrs=['bold']), end='\r')
    self.__iterateDataset(self.__writeImages)
    self.__writeLabels()
    self.__clearTmp()
    print(end='\x1b[2K')
    print(colored(' Ready           💚', 'green', attrs=['bold']))

  def __downloadDataset(self):
    if not os.path.exists(self.__tmp_path):
      os.mkdir(self.__tmp_path)

    with contextlib.redirect_stdout(io.StringIO()):
      with contextlib.redirect_stderr(io.StringIO()):
        if self.datasetName == 'MNIST':
          self.__dataset = torchvision.datasets.MNIST(self.__tmp_path, download=True)
        elif self.datasetName == 'FashionMNIST':
          self.__dataset = torchvision.datasets.FashionMNIST(self.__tmp_path, download=True)

  def __clearTmp(self):
    filelist = glob.glob(f"{self.__tmp_path}/{self.datasetName}/raw/*")
    for f in filelist:
      os.remove(f)
    os.rmdir(f"{self.__tmp_path}/{self.datasetName}/raw")
    os.rmdir(f"{self.__tmp_path}/{self.datasetName}")

  def __iterateDataset(self, cb_fn):
    idx = 0
    max_train_idx = int(len(self.__dataset) * self.split_percentage / 100)

    for img, label in self.__dataset:
      label = self.__customLabelMapping(label)

      if idx < max_train_idx:
        cb_fn(img, idx, self.train_dir)
        self.__train_labels.append(f"{idx},{label}")
      else:
        cb_fn(img, idx, self.test_dir)
        self.__test_labels.append(f"{idx},{label}")

      idx+=1

  def __writeImages(self, img, idx, path):
    img.save(f"{path}/{idx}.jpg")

  def __customLabelMapping(self, label):
    if self.datasetName == 'FashionMNIST':
      return [
        'Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
      ][label]
    else:
      return label

  def __writeLabels(self):
    with open(f'{self.train_dir}/labels.csv', 'w') as f:
      for label in self.__train_labels:
        f.write(label)
        f.write('\n')
    f.close()

    with open(f'{self.test_dir}/labels.csv', 'w') as f:
      for label in self.__test_labels:
        f.write(label)
        f.write('\n')
    f.close()

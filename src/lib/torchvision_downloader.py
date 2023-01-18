import torchvision
import glob, os, os.path, contextlib, io
from termcolor import colored

class TorchvisionDownloader(object):
  def __init__(self, image_dir = 'data/test', dataset = 'MNIST'):
    self.datasetName = dataset
    self.__dataset = None
    self.__tmp_path = f"tmp"
    self.image_dir = image_dir
    self.__labels = []

  def call(self):
    if not os.path.exists(self.__tmp_path) or not os.path.exists(f"{self.__tmp_path}/{self.datasetName}"):
      print(colored(' Downloading ... 🌑 ', 'blue', attrs=['bold']))
      self.__downloadDataset()
    else:
      self.__dataset = torchvision.datasets.MNIST(self.__tmp_path)

    print(colored(' Processing ...  🌕 ', 'yellow', attrs=['bold']), end='\r')
    self.__iterateDataset(self.__writeImages)
    self.__writeLabels()
    self.__clearTmp()
    print(end='\x1b[2K')
    print(colored(' Ready           💚', 'green', attrs=['bold']))

  def __downloadDataset(self):
    if not os.path.exists(self.__tmp_path):
      os.mkdir(self.__tmp_path)
    if self.datasetName == 'MNIST':
      with contextlib.redirect_stdout(io.StringIO()):
        self.__dataset = torchvision.datasets.MNIST(self.__tmp_path, download=True)

  def __clearTmp(self):
    filelist = glob.glob(f"{self.__tmp_path}/{self.datasetName}/raw/*")
    for f in filelist:
      os.remove(f)
    os.rmdir(f"{self.__tmp_path}/{self.datasetName}/raw")
    os.rmdir(f"{self.__tmp_path}/{self.datasetName}")

  def __iterateDataset(self, cb_fn):
    idx = 0
    for img, label in self.__dataset:
      cb_fn(img, idx)
      self.__labels.append(f"{idx},{label}")
      idx+=1

  def __writeImages(self, img, idx):
    img.save(f"{self.image_dir}/{idx}.jpg")

  def __writeLabels(self):
    with open(f'{self.image_dir}/labels.csv', 'w') as f:
      for label in self.__labels:
        f.write(label)
        f.write('\n')
    f.close()

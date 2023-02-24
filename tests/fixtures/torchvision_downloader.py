import pytest; from unittest.mock import MagicMock
from features import TorchvisionDownloader
from dependencies.core import tempfile, os

CLASS_TO_IDX = {"Dress": 0, "Shirt": 1, "Sneaker": 2, "T-Shirt": 3}

class MockImage:
  def __init__(self):
    pass
  def save(self, filepath):
    with open(filepath, 'w') as f:
      f.write('')

class DatasetMock:
  def __init__(self, class_to_idx = CLASS_TO_IDX):
    if class_to_idx is not None:
      self.class_to_idx = class_to_idx

    self.__imgs = [
      (MockImage(), 0),
      (MockImage(), 1),
      (MockImage(), 2),
      (MockImage(), 3),
    ]

  def __iter__(self):
    return iter(self.__imgs)

  def __len__(self):
    return len(self.__imgs)

@pytest.fixture
def bad_train_directory_fixture():
  f = tempfile.mkdtemp(prefix='alicia-test-bad-')
  os.mkdir(f"{f}/train")
  return f

@pytest.fixture
def bad_test_directory_fixture():
  f = tempfile.mkdtemp(prefix='alicia-test-bad-')
  os.mkdir(f"{f}/test")
  return f

@pytest.fixture
def bad_valid_directory_fixture():
  f = tempfile.mkdtemp(prefix='alicia-test-bad-')
  os.mkdir(f"{f}/valid")
  return f

@pytest.fixture
def tmp_directory_fixture():
  return tempfile.mkdtemp(prefix="alicia-test")

@pytest.fixture
def torchvision_downloader_fixture(tmp_directory_fixture):
  return TorchvisionDownloader(tmp_directory_fixture, "FakeData",
    size = 24, image_size = (1,1,1), num_classes = 3
  )

@pytest.fixture
def torchvision_downloader_no_dir_fixture(tmp_directory_fixture):
  os.rmdir(tmp_directory_fixture)
  return TorchvisionDownloader(tmp_directory_fixture, "FakeData",
    size = 24, image_size = (1,1,1), num_classes = 3
  )

@pytest.fixture
def class_to_idx_fixture():
  return CLASS_TO_IDX

@pytest.fixture
def categories_fixture():
  return { 0: "T-Shirt", 1: "Sneaker", 2: "Shirt", 3: "Dress" }

@pytest.fixture
def torchvision_downloader_mocked_fixture(tmp_directory_fixture):
  d = TorchvisionDownloader(tmp_directory_fixture, "MNIST", train = False)
  d._get_all_posible_splits = MagicMock(return_value=[DatasetMock()])
  return d

@pytest.fixture
def torchvision_downloader_mocked_bad_category_fixture(tmp_directory_fixture):
  d = TorchvisionDownloader(tmp_directory_fixture, "MNIST", train = False)
  class_to_idx = {"Dress": "0", "Shirt": "1", "Sneaker": "2", "T-Shirt": "3"}
  # class_to_idx = { "0": "T-Shirt", "1": "Sneaker", "2": "Shirt", "3": "Dress" }
  d._get_all_posible_splits = MagicMock(return_value=[DatasetMock(class_to_idx = class_to_idx)])
  return d

@pytest.fixture
def torchvision_downloader_mocked_w_categories_fixture(tmp_directory_fixture):
  d = TorchvisionDownloader(tmp_directory_fixture, "MNIST", train = False)
  d._get_all_posible_splits = MagicMock(return_value=[DatasetMock(class_to_idx = None)])
  return d

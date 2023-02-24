import pytest;
from features import TorchvisionDownloader
from tests.fixtures.torchvision_downloader import *

class TestTorchvisionDownloader:
  """
    Notes:
    ------
      * Using [FakeData](https://pytorch.org/vision/0.8/datasets.html#fakedata) as download target
  """
  def test_initialization(self, torchvision_downloader_fixture):
    d = torchvision_downloader_fixture
    assert isinstance(d.dataset_name, str)
    assert isinstance(d.split_percentage, tuple)
    assert d.dataset_name == "FakeData"

  def test_fail_to_download_a_bad_dataset(self, torchvision_downloader_fixture):
    with pytest.raises(Exception):
      torchvision_downloader_fixture.call()

  def test_initialization_with_existing_train_dir(
      self, bad_train_directory_fixture, bad_valid_directory_fixture, bad_test_directory_fixture
  ):
    with pytest.raises(Exception):
      _ = TorchvisionDownloader(bad_train_directory_fixture, "FakeData")
    with pytest.raises(Exception):
      _ = TorchvisionDownloader(bad_valid_directory_fixture, "FakeData")
    with pytest.raises(Exception):
      _ = TorchvisionDownloader(bad_test_directory_fixture, "FakeData")

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

  @pytest.mark.download
  def test_call_create_images_in_the_three_folders(
      self, torchvision_downloader_really_fixture, tmp_directory_fixture
  ):
    d = torchvision_downloader_really_fixture
    d.call()
    assert len(os.listdir(f"{tmp_directory_fixture}/train")) > 0
    assert len(os.listdir(f"{tmp_directory_fixture}/valid")) > 0
    assert len(os.listdir(f"{tmp_directory_fixture}/test")) > 0

  @pytest.mark.download
  def test_call_create_labels_file(
      self, torchvision_downloader_really_fixture, tmp_directory_fixture
  ):
    d = torchvision_downloader_really_fixture
    d.call()
    assert os.path.isfile(f"{tmp_directory_fixture}/labels.csv")


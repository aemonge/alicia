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

  def should_create_directory_if_not_exists(self, torchvision_downloader_no_dir_fixture, tmp_directory_fixture):
    _ = torchvision_downloader_no_dir_fixture
    assert os.path.exists(tmp_directory_fixture)
    assert os.path.exists(f"{tmp_directory_fixture}/train")
    assert os.path.exists(f"{tmp_directory_fixture}/valid")
    assert os.path.exists(f"{tmp_directory_fixture}/test")

  def should_fail_to_download_a_bad_dataset(self, torchvision_downloader_fixture):
    with pytest.raises(Exception):
      torchvision_downloader_fixture.call()

  def should_initialization_with_existing_train_dir(
      self, bad_train_directory_fixture, bad_valid_directory_fixture, bad_test_directory_fixture
  ):
    with pytest.raises(Exception):
      _ = TorchvisionDownloader(bad_train_directory_fixture, "FakeData")
    with pytest.raises(Exception):
      _ = TorchvisionDownloader(bad_valid_directory_fixture, "FakeData")
    with pytest.raises(Exception):
      _ = TorchvisionDownloader(bad_test_directory_fixture, "FakeData")

  def should_call_create_images_in_the_three_folders(
      self, torchvision_downloader_mocked_fixture, tmp_directory_fixture
  ):
    d = torchvision_downloader_mocked_fixture
    d.call()
    assert len(os.listdir(f"{tmp_directory_fixture}/train")) > 0
    assert len(os.listdir(f"{tmp_directory_fixture}/valid")) > 0
    assert len(os.listdir(f"{tmp_directory_fixture}/test")) > 0

  def should_call_create_labels_file(
      self, torchvision_downloader_mocked_fixture, tmp_directory_fixture
  ):
    d = torchvision_downloader_mocked_fixture
    d.call()
    assert os.path.isfile(f"{tmp_directory_fixture}/labels.csv")

  def should_use_categories_as_labels(
      self, torchvision_downloader_mocked_w_categories_fixture, categories_fixture
  ):
    d = torchvision_downloader_mocked_w_categories_fixture
    d.call(categories = categories_fixture)
    assert d._idx_to_class == categories_fixture

  def should_fail_if_no_use_categories_as_labels(
      self, torchvision_downloader_mocked_w_categories_fixture
  ):
    d = torchvision_downloader_mocked_w_categories_fixture
    with pytest.raises(Exception):
      d.call()

  def should_continue_with_broken_categories_as_labels(
      self, torchvision_downloader_mocked_bad_category_fixture
  ):
    d = torchvision_downloader_mocked_bad_category_fixture
    d.call()
    assert len(d._labels) > 0

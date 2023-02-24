from tests.fixtures.torchvision_downloader import *

class E2ETorchvisionDownloader:
  def test_call_create_images_in_the_three_folders(
      self, torchvision_downloader_really_fixture, tmp_directory_fixture
  ):
    d = torchvision_downloader_really_fixture
    d.call()
    assert len(os.listdir(f"{tmp_directory_fixture}/train")) > 0
    assert len(os.listdir(f"{tmp_directory_fixture}/valid")) > 0
    assert len(os.listdir(f"{tmp_directory_fixture}/test")) > 0

  def test_call_create_labels_file(
      self, torchvision_downloader_really_fixture, tmp_directory_fixture
  ):
    d = torchvision_downloader_really_fixture
    d.call()
    assert os.path.isfile(f"{tmp_directory_fixture}/labels.csv")


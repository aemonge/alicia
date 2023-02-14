import pytest;
from features import TorchvisionDownloader
from dependencies.core import tempfile, os

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
def torchvision_downloader_really_fixture(tmp_directory_fixture):
  return TorchvisionDownloader(tmp_directory_fixture, "MNIST", train = False)


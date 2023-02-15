import pytest; from unittest.mock import MagicMock # , ANY
from modules.models import Basic
from features import Trainer
from dependencies.core import csv, pathlib, torchvision, Image

def mock_all_prints(t):
  t.__get_step_color__ = MagicMock()
  t._loading = MagicMock()
  t._spin = MagicMock()
  t.__backspace__ = MagicMock()
  t._print_train_header = MagicMock()
  t._print_test_header = MagicMock()
  t._print_step_header = MagicMock()
  t._print_step = MagicMock()
  t._print_total = MagicMock()
  t._print_t_step = MagicMock()
  t._print_pbs = MagicMock()
  t._imshow = MagicMock()
  return t

@pytest.fixture
def data_tmp_dir_fixture():
  return pathlib.Path.cwd().joinpath("tests/fixtures/data")

@pytest.fixture
def data_shirt_image_fixture():
  img = Image.open(pathlib.Path.cwd().joinpath("tests/fixtures/data/test/35921.jpg"))

  return img

@pytest.fixture
def data_tmp_dir_labels_fixture():

  return pathlib.Path.cwd().joinpath("tests/fixtures/data/labels.csv")

@pytest.fixture
def labels_fixture():
  return [
    'Ankle boot', 'Bag', 'Coat', 'Dress', 'Pullover', 'Sandal', 'Shirt', 'Sneaker', 'Top', 'Trouser',
  ]

@pytest.fixture
def labels_dict_fixture(data_tmp_dir_labels_fixture):
  labels = {}
  with open(data_tmp_dir_labels_fixture, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for filename, label in reader:
      labels[filename] = label
  return labels

@pytest.fixture
def model_fixture(labels_fixture):
  b = Basic(labels_fixture)
  b.create()

  return b

@pytest.fixture
def transforms_fixture():
  t = torchvision.transforms.Compose([ torchvision.transforms.ToTensor() ])
  return { "valid": t, "display": t, "test": t, "train": t }

@pytest.fixture
def trainer_fixture(model_fixture, transforms_fixture):
  t = Trainer(model_fixture, transforms_fixture)

  return mock_all_prints(t)

@pytest.fixture
def trainer_with_momentum_fixture(model_fixture, transforms_fixture):
  t = Trainer(model_fixture, transforms_fixture, momentum = 0.9)

  return mock_all_prints(t)

@pytest.fixture
def trainer_with_big_momentum_fixture(model_fixture, transforms_fixture):
  t = Trainer(model_fixture, transforms_fixture, momentum = 2)

  return mock_all_prints(t)

@pytest.fixture
def trainer_with_bad_transforms_fixture(model_fixture):
  t = Trainer(model_fixture, {})

  return mock_all_prints(t)
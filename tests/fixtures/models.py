import pytest;
from dependencies.core import csv, pathlib
from modules.models import Basic

@pytest.fixture
def labels_fixture():
  return [
    'Ankle boot', 'Bag', 'Coat', 'Dress', 'Pullover', 'Sandal', 'Shirt', 'Sneaker', 'Top', 'Trouser',
  ]

@pytest.fixture
def data_tmp_dir_labels_fixture():
  return pathlib.Path.cwd().joinpath("tests/fixtures/data/labels.csv")

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

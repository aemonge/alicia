import pytest;
from dependencies.core import csv, pathlib
from modules.models import Elemental

@pytest.fixture
def labels_fixture():
  return [
    '0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six',
    '7 - seven', '8 - eight', '9 - nine'
  ]

@pytest.fixture
def data_tmp_main_dir_fixture():
  return "tests/fixtures/data"

@pytest.fixture
def data_tmp_dir_fixture(data_tmp_main_dir_fixture):
  data_dir = {
      'train':
      pathlib.Path.cwd().joinpath(f"{data_tmp_main_dir_fixture}/train")
  }
  data_dir['test'] = pathlib.Path.cwd().joinpath(
    f"{data_tmp_main_dir_fixture}/test"
  )
  data_dir['valid'] = pathlib.Path.cwd().joinpath(
    f"{data_tmp_main_dir_fixture}/valid"
  )

  return data_dir
  # return pathlib.Path.cwd().joinpath("tests/fixtures/data/")

@pytest.fixture
def data_tmp_dir_labels_fixture():
  return pathlib.Path.cwd().joinpath("tests/fixtures/data/mnist-labels.csv").as_posix()

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
  return Elemental(init_features=True, labels=labels_fixture, input_size=(28 * 28))

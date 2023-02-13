import pytest
import csv
from unittest.mock import MagicMock, ANY
import pathlib as Pathlib

# My modules
from features.trainer.Trainer import Trainer
from modules.models.Basic import Basic

import torch
# from torch.utils.data import criterion, transforms
from torchvision import transforms as Transforms

# Types
from torchvision.transforms.transforms import Compose

# Constants
BATCH_SIZE = 4
TRAIN_COUNT = 24

@pytest.fixture
def data_tmp_dir_fixture():
  return Pathlib.Path.cwd().joinpath("tests/fixtures/data")

@pytest.fixture
def data_tmp_dir_labels_fixture():

  return Pathlib.Path.cwd().joinpath("tests/fixtures/data/labels.csv")

@pytest.fixture
def labels_fixture():
  return ['Ankle boot', 'Bag', 'Coat', 'Dress', 'Pullover', 'Sandal', 'Shirt', 'Sneaker', 'Top', 'Trouser']

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
  t = Transforms.Compose([ Transforms.ToTensor() ])
  return { "valid": t, "display": t, "test": t, "train": t }

@pytest.fixture
def trainer_fixture(model_fixture, transforms_fixture):
  t = Trainer(model_fixture, transforms_fixture)

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

class TestTrainer:
  @pytest.fixture(autouse=True)
  def test_init(self, trainer_fixture):
    """
      Main attributes should exists.
    """
    assert isinstance(trainer_fixture, Trainer)
    assert isinstance(trainer_fixture.model, Basic)
    assert isinstance(trainer_fixture.learning_rate, float)
    assert isinstance(trainer_fixture.transforms, dict)
    assert isinstance(trainer_fixture.transforms["valid"], Compose)
    assert isinstance(trainer_fixture.transforms["train"], Compose)
    assert isinstance(trainer_fixture.transforms["test"], Compose)
    assert isinstance(trainer_fixture.criterion, torch.nn.modules.loss._Loss)
    assert isinstance(trainer_fixture.optimizer, torch.optim.Optimizer)

  def test_train_should_call_trian_step(self, trainer_fixture, labels_dict_fixture, data_tmp_dir_fixture):
    t = trainer_fixture
    t.train_step = MagicMock(return_value=0.105)
    t.train(data_tmp_dir_fixture, labels_dict_fixture, BATCH_SIZE, 1)
    t.train_step.assert_called()

  def test_train_should_call_validation_step(self, trainer_fixture, labels_dict_fixture, data_tmp_dir_fixture):
    t = trainer_fixture
    t.validation_step = MagicMock(return_value=(0.105, 30))
    t.train(data_tmp_dir_fixture, labels_dict_fixture, BATCH_SIZE, 1)
    t.validation_step.assert_called_once()

  def test_train_should_call_train_step_same_times_as_batches(
      self, trainer_fixture, labels_dict_fixture, data_tmp_dir_fixture
  ):
    call_count = round(TRAIN_COUNT / BATCH_SIZE) # 700 Is the file size of /train
    t = trainer_fixture
    t.train_step = MagicMock(return_value=0.105)
    t.train(data_tmp_dir_fixture, labels_dict_fixture, BATCH_SIZE, 1)
    assert t.train_step.call_count == call_count

  def test_train_should_call_validation_step_same_times_as_epochs(
      self, trainer_fixture, labels_dict_fixture, data_tmp_dir_fixture
  ):
    t = trainer_fixture
    t.validation_step = MagicMock(return_value=(0.105, 30))
    t.train(data_tmp_dir_fixture, labels_dict_fixture, batch_size = BATCH_SIZE, epochs = 3)
    assert t.validation_step.call_count == 3

  def test_train_should_freeze_parameters(
      self, trainer_fixture, labels_dict_fixture, data_tmp_dir_fixture
  ):
    t = trainer_fixture
    t.train_step = MagicMock(return_value=0.105)
    t.train(data_tmp_dir_fixture, labels_dict_fixture, BATCH_SIZE, 1, freeze_parameters = True)
    for param in t.model.features.parameters():
      assert param.requires_grad == False

  def test_train_should_call_the_print_total_fn(
      self, trainer_fixture, labels_dict_fixture, data_tmp_dir_fixture
  ):
    t = trainer_fixture
    t._print_total = MagicMock()
    t.train(data_tmp_dir_fixture, labels_dict_fixture)
    t._print_total.assert_called_once()

  def test_train_loss_should_get_lower(
      self, trainer_fixture, labels_dict_fixture, data_tmp_dir_fixture
  ):
    t = trainer_fixture
    t.train_step = MagicMock(side_effect=t.train_step)
    t.train(data_tmp_dir_fixture, labels_dict_fixture, BATCH_SIZE, 1)

    assert t.train_step.mock_calls[0].args[2] == 0.0 # assert_called_with(ANY, ANY, 0.0)

    last_loss = t.train_step.mock_calls[1].args[2]
    for call in t.train_step.mock_calls[2:]:
      current_loss = call.args[2]
      assert last_loss >= current_loss
      last_loss = current_loss

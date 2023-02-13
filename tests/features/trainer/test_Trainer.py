import pytest; from unittest.mock import MagicMock # , ANY
from dependencies.core import torch, transforms
from features import Trainer
from modules.models import Basic
from tests.fixtures.trainer import *

# Constants
BATCH_SIZE = 4
TRAIN_COUNT = 24

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
    assert isinstance(trainer_fixture.transforms["valid"], transforms.transforms.Compose)
    assert isinstance(trainer_fixture.transforms["train"], transforms.transforms.Compose)
    assert isinstance(trainer_fixture.transforms["test"], transforms.transforms.Compose)
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

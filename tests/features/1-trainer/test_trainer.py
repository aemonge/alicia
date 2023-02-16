from dependencies.core import torch, torchvision
from features import Trainer
from modules.models import Elemental
from tests.fixtures.trainer import *

# Constants
BATCH_SIZE = 4
TRAIN_COUNT = 24
TEST_BATCH = 2
TEST_COUNT = 4

class TestTrainer:
  def test_init(self, trainer_fixture):
    assert isinstance(trainer_fixture, Trainer)
    assert isinstance(trainer_fixture.model, Elemental)
    assert isinstance(trainer_fixture.learning_rate, float)
    assert isinstance(trainer_fixture.transforms, dict)
    assert isinstance(trainer_fixture.transforms["valid"], torchvision.transforms.transforms.Compose)
    assert isinstance(trainer_fixture.transforms["train"], torchvision.transforms.transforms.Compose)
    assert isinstance(trainer_fixture.transforms["test"], torchvision.transforms.transforms.Compose)
    assert isinstance(trainer_fixture.criterion, torch.nn.modules.loss._Loss)
    assert isinstance(trainer_fixture.optimizer, torch.optim.Optimizer)

  def test_init_with_momentum(self, trainer_with_momentum_fixture):
    assert isinstance(trainer_with_momentum_fixture.momentum, float)

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
    call_count = round(TRAIN_COUNT / BATCH_SIZE)
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

  # @pytest.mark.xfail(reason="Known issue, waiting for other models to do a better test on this")
  def test_train_loss_should_get_lower(
      self, trainer_fixture, labels_dict_fixture, data_tmp_dir_fixture
  ):
    t = trainer_fixture
    t.train_step = MagicMock(side_effect=t.train_step)
    t.train(data_tmp_dir_fixture, labels_dict_fixture, BATCH_SIZE, 1)

    assert t.train_step.mock_calls[0].args[2] == 0.0

    calls = t.train_step.mock_calls
    calls.reverse() # Since the call stack is sorted in the other way ;)

    last_loss = calls[1].args[2]
    for call in t.train_step.mock_calls[2:]:
      current_loss = call.args[2]
      assert last_loss >= current_loss
      last_loss = current_loss

  def test_raising_exepction_on_bad_transformers_by_training(self, trainer_with_bad_transforms_fixture):
    with pytest.raises(ValueError):
      t = trainer_with_bad_transforms_fixture
      t.train_step = MagicMock(return_value=0.105)
      t.train(data_tmp_dir_fixture, labels_dict_fixture, BATCH_SIZE, 1)

  def test_raising_exepction_on_bad_transformers_by_testing(self, trainer_with_bad_transforms_fixture):
    with pytest.raises(ValueError):
      t = trainer_with_bad_transforms_fixture
      t.test(data_tmp_dir_fixture, labels_dict_fixture, BATCH_SIZE, 1)

  def test_raising_exepction_on_big_momentum(self, trainer_with_big_momentum_fixture):
    with pytest.raises(Exception):
      t = trainer_with_big_momentum_fixture
      t.train(data_tmp_dir_fixture, labels_dict_fixture, BATCH_SIZE, 1)

  def test_raising_exepction_on_when_loss_is_nan(self, trainer_fixture):
    with pytest.raises(Exception):
      t = trainer_fixture
      t.train_step = MagicMock(return_value=math.nan)
      t.train(data_tmp_dir_fixture, labels_dict_fixture, BATCH_SIZE, 1)

  def test_should_call_the_model_batch_times(self, trainer_fixture, labels_dict_fixture, data_tmp_dir_fixture):
    call_count = round(TEST_COUNT / TEST_BATCH)
    t = trainer_fixture
    t.model.forward = MagicMock(side_effect=t.model.forward)
    t.test(data_tmp_dir_fixture, labels_dict_fixture, batch_size = TEST_BATCH)

    assert t.model.forward.call_count == call_count

  def test_should_call_print_t_step(self, trainer_fixture, labels_dict_fixture, data_tmp_dir_fixture):
    t = trainer_fixture
    t.test(data_tmp_dir_fixture, labels_dict_fixture, batch_size = TEST_BATCH)
    t._print_t_step.assert_called()

  def test_test_should_freeze_parameters(
      self, trainer_fixture, labels_dict_fixture, data_tmp_dir_fixture
  ):
    t = trainer_fixture
    t.test(data_tmp_dir_fixture, labels_dict_fixture, batch_size = TEST_BATCH, freeze_parameters = True)
    for param in t.model.features.parameters():
      assert param.requires_grad == False

  def test_predict_should_return_a_single_prediction(
      self, trainer_fixture, data_shirt_image_fixture
  ):
    t = trainer_fixture
    ps_val, ps_id = t.predict(data_shirt_image_fixture, 1)
    assert len(ps_val) == 1
    assert len(ps_id) == 1

  def test_predict_should_return_four_predictions(
      self, trainer_fixture, data_shirt_image_fixture
  ):
    t = trainer_fixture
    ps_val, ps_id = t.predict(data_shirt_image_fixture, 4)
    assert len(ps_val) == 4
    assert len(ps_id) == 4

  def test_predict_should_return_some_predictions(
      self, trainer_fixture, data_shirt_image_fixture
  ):
    t = trainer_fixture
    ps_val, ps_id = t.predict(data_shirt_image_fixture, 2)
    assert isinstance(ps_id[0], str)
    assert isinstance(ps_id[1], str)
    assert ps_val[0] > 0.0
    assert ps_val[1] > 0.0

  def test_predict_image_sholud_call_predict(
      self, trainer_fixture, path_shirt_image_fixture
  ):
    t = trainer_fixture
    t.predict = MagicMock()
    t.predict_image(path_shirt_image_fixture)
    t.predict.assert_called()

  def test_predict_image_sholud_call_predict_with_args_and_kwargs(
      self, trainer_fixture, data_shirt_image_fixture, path_shirt_image_fixture
  ):
    t = trainer_fixture
    t.predict = MagicMock()
    t.predict_image(path_shirt_image_fixture, topk=2)
    t.predict.assert_called_with(data_shirt_image_fixture, topk=2)

import pytest
from dependencies.datatypes import *
from features import Comparer
from tests.fixtures.comparer import *

class TestComparer:
  def test_init(self, comparer_fixture):
    assert isinstance(comparer_fixture, Comparer)
    assert isinstance(comparer_fixture.models, list)
    assert isinstance(comparer_fixture.models[0], AbsModule)
    assert isinstance(comparer_fixture.names, list)
    assert isinstance(comparer_fixture.names[0], str)

  def test_accuracy_for_two_should_call_twice_test_from_trainer(
    self, comparer_fixture, data_tmp_dir_fixture, data_tmp_dir_labels_fixture
  ):
    c = comparer_fixture
    c.accuracy(data_tmp_dir_fixture, data_tmp_dir_labels_fixture, 8)
    assert c.Trainer.test.call_count == 2

  def test_accuracy_for_three_should_call_trice_test_from_trainer(
    self, comparer_3_fixture, data_tmp_dir_fixture, data_tmp_dir_labels_fixture
  ):
    c = comparer_3_fixture
    c.accuracy(data_tmp_dir_fixture, data_tmp_dir_labels_fixture, 1)
    assert c.Trainer.test.call_count == 3

  def test_accuracy_should_call_test_from_trainer_with_args_and_kwargs(
    self, comparer_fixture, data_tmp_dir_fixture, data_tmp_dir_labels_fixture
  ):
    c = comparer_fixture
    c.accuracy(data_tmp_dir_fixture, data_tmp_dir_labels_fixture, 1)
    c.Trainer.test.has_been_called_with(data_tmp_dir_fixture, data_tmp_dir_labels_fixture, 1)
    c.accuracy(data_tmp_dir_fixture, data_tmp_dir_labels_fixture, 42, demo='ok')
    c.Trainer.test.has_been_called_with(data_tmp_dir_fixture, data_tmp_dir_labels_fixture, 42, demo='ok')

  def test_accuracy_should_call_print_results(
    self, comparer_fixture, data_tmp_dir_fixture, data_tmp_dir_labels_fixture
  ):
    c = comparer_fixture
    c.accuracy(data_tmp_dir_fixture, data_tmp_dir_labels_fixture)
    c._print_results.has_been_called

  def test_training_comparing_two_should_call_twice_train(
    self, comparer_fixture, data_tmp_dir_fixture, data_tmp_dir_labels_fixture
  ):
    c = comparer_fixture
    c.training(data_tmp_dir_fixture, data_tmp_dir_labels_fixture, 1)
    assert c.Trainer.train.call_count == 2

  def test_training_comparing_three_should_call_trice_train(
    self, comparer_3_fixture, data_tmp_dir_fixture, data_tmp_dir_labels_fixture
  ):
    c = comparer_3_fixture
    c.training(data_tmp_dir_fixture, data_tmp_dir_labels_fixture, 1)
    assert c.Trainer.train.call_count == 3

  def test_training_should_call_train_from_trainer_with_one_epoch_only(
    self, comparer_fixture, data_tmp_dir_fixture, data_tmp_dir_labels_fixture
  ):
    c = comparer_fixture
    c.training(data_tmp_dir_fixture, data_tmp_dir_labels_fixture, 16)
    c.Trainer.train.has_been_called_with(data_tmp_dir_fixture, data_tmp_dir_labels_fixture, 1, 16)

  def test_training_should_call_train_from_trainer_with_kwargs(
    self, comparer_fixture, data_tmp_dir_fixture, data_tmp_dir_labels_fixture
  ):
    c = comparer_fixture
    c.training(data_tmp_dir_fixture, data_tmp_dir_labels_fixture, 8, demo='ok')
    c.Trainer.train.has_been_called_with(data_tmp_dir_fixture, data_tmp_dir_labels_fixture, 1, 8, demo='ok')

  def test_when_train_raises_exception_should_raise_exception_and_terminate_loading(
    self, comparer_fixture, data_tmp_dir_fixture, data_tmp_dir_labels_fixture
  ):
    c = comparer_fixture
    c.Trainer.train.side_effect = Exception
    with pytest.raises(Exception):
      c.training(data_tmp_dir_fixture, data_tmp_dir_labels_fixture)
    c._terminate_loading.assert_called_once

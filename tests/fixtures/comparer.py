import pytest; # from unittest.mock import MagicMock # , ANY
from features import Comparer
from fixtures.models import *

@pytest.fixture
def models_fixture(model_fixture):
  models = [model_fixture, model_fixture]
  for model in models:
    model.load()
  return

@pytest.fixture
def models_names_fixture():
  return ['test-model-1.pth', 'test-model-2.pth', 'test-model-3.pth']

@pytest.fixture
def comparer_fixture(models_fixture, models_names_fixture):
  c = Comparer(models_fixture, names_fixture=models_names_fixture)

  return c

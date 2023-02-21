from dependencies.core import torch
import pytest; from unittest.mock import MagicMock # , ANY
from features import Comparer, Trainer
from fixtures.models import *

def mock_prints(c):
  c._loading = MagicMock()
  c._terminate_loading = MagicMock()
  c._print_results = MagicMock()
  return c

@pytest.fixture
def models_fixture(model_fixture, models_names_fixture):
  models = [model_fixture, model_fixture, model_fixture]
  for model, file in list(zip(models, models_names_fixture)):
    data = torch.load(f"tests/fixtures/{file}")
    model.load(data['state_dict'])
  return models

@pytest.fixture
def models_names_fixture():
  return ['elemental-five-epochs.pth', 'elemental-not-trained.pth', 'elemental-two-epochs.pth']

@pytest.fixture
def comparer_fixture(models_fixture, models_names_fixture):
  c = Comparer(Trainer, models_fixture[:2], names = models_names_fixture)
  c.Trainer.test = MagicMock()
  c.Trainer.train = MagicMock()
  return mock_prints(c)

@pytest.fixture
def comparer_3_fixture(models_fixture, models_names_fixture):
  c = Comparer(Trainer, models_fixture, names = models_names_fixture)
  c.Trainer.test = MagicMock()
  c.Trainer.train = MagicMock()
  return mock_prints(c)

@pytest.fixture
def transforms_fixture():
  t = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    Reshapetransform((-1, 28*28)),
  ])
  return { "valid": t, "display": t, "test": t, "train": t }

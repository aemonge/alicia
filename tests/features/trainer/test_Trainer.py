from features.trainer.Trainer import Trainer
from modules.models.Basic import Basic

# from torch.utils.data import criterion
from torchvision import transforms as Transforms

t = Transforms.Compose([Transforms.ToTensor()])
transforms = { "valid": t, "display": t, "test": t, "train": t }

class TestTrainer:
  pokemos = ['pikachu', 'bulbasaur', 'charmander', 'squirtle']
  # model = Basic(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'])
  # trainer = Trainer(model, transforms)

  def test_one(self):
    """
      Does this help at all?
    """
    assert isinstance(self.pokemos, list)

  def test_should_fail(self):
    """
      Does this help at all?
    """
    assert isinstance(self.pokemos, dict)

  # def test_two(self):
  #   """
  #     Be more verbose
  #   """
  #   assert isinstance(self.model, Basic)

  # def should_initialize_the_trainer(self):
  #   assert isinstance(self.trainer, Trainer)
    # assert isinstance(self.trainer.criterion,criterion)

# class TrainerTest:
#   trainer = Trainer(model: AbsModule, transforms)
#
#   def train_should_succeed(self):
#     assert 1 + 1 == 2

# t = TrainerInit()
# t.should_initialize_the_trainer()

import torch
from abc import abstractmethod, ABCMeta

class AbsModule(torch.nn.Module, metaclass=ABCMeta):
  # labels : list # TODO: Define this property as required in the class look the docs for ABC

  @abstractmethod
  def create(self):
    pass

  @abstractmethod
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    pass

  # TODO: Do you really need to re-implement this? I would guess not bro
  # @abstractmethod
  # def eval(self):
  #   pass
  #
  # @abstractmethod
  # def train(self):
  #   pass

  @abstractmethod
  def parameters(self) -> list:
    pass

  @abstractmethod
  def save(self, path: str):
    pass

  @abstractmethod
  def load(self, path: str):
    pass

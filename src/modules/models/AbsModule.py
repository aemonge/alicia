import torch
from abc import abstractmethod
from better_abc import abstract_attribute, ABCMeta

class AbsModule(torch.nn.Module, metaclass=ABCMeta):
  @abstract_attribute
  def labels(self) -> list|None:
    pass

  # @abstract_attribute
  # def features(self):
  #   pass

  @abstractmethod
  def create(self):
    pass

  @abstractmethod
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    pass

  @abstractmethod
  def parameters(self) -> list:
    pass

  @abstractmethod
  def save(self, path: str):
    pass

  @abstractmethod
  def load(self, path: str):
    pass

import torch
import abc
from PIL import Image
from abc import abstractmethod

class Module(torch.nn.Module, metaclass=abc.ABCMeta):
  optimizer : torch.optim.Optimizer
  transforms : dict

  @abstractmethod
  def create(self):
    pass

  @abstractmethod
  def eval(self):
    pass

  @abstractmethod
  def train(self):
    pass

  @abstractmethod
  def save(self, path: str):
    pass

  @abstractmethod
  def load(self, path: str):
    pass

from dependencies.core import time, random, torch
from dependencies.datatypes import Iterator, Parameter
from .abs_module import AbsModule

class Dummy(AbsModule):
  """
    A dummy module that return random sleep time, simple to test graphical output.
  """
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    return self.forward(x)

  def __repr__(self):
    return 'Dummy()'

  def __init__(self, _) -> None:
    super().__init__()
    self.num_classes = 5

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    time.sleep(random.uniform(0.0, 2.5))
    return torch.randint(0, self.num_classes, (x.shape[0],))

  def load(self) -> None:
    raise NotImplementedError("A dummy is not expected to be used production")

  def parameters(self) -> Iterator[Parameter]:
    s = torch.nn.Sequential(torch.nn.Linear(10, self.num_classes))
    return s.parameters()

  def save(self) -> None:
    raise NotImplementedError("A dummy is not expected to be used production")

  def create(self) -> None:
    pass

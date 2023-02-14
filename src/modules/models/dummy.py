from dependencies.core import time, random, torch
from dependencies.datatypes import Iterator, Parameter
from .abs_module import AbsModule

class Dummy(AbsModule):
  """
    A dummy module that return random sleep time, simple to test graphical output.

    Methods:
    --------
      forward(self, x: torch.Tensor) -> torch.Tensor
        A forward pass of the neural network.
  """
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    """
      A forward pass of the neural network.

      Parameters:
      -----------
        x: torch.Tensor
          A batch of input features.

      Returns:
      --------
        torch.Tensor
    """
    return self.forward(x)

  def __repr__(self):
    """
      A string representation of the neural network.

      Returns:
      --------
        : str
          A string representation 'Dummy()'.
    """
    return 'Dummy()'

  def __init__(self, _) -> None:
    """
      Constructor of the neural network.

      Parameters:
      -----------
    """
    super().__init__()
    self.num_classes = 5

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
      A forward pass of the neural network.

      Parameters:
      -----------
        x: torch.Tensor
          A batch of input features.

      Returns:
      --------
        torch.Tensor
          Random tensor.
    """
    time.sleep(random.uniform(0.0, 2.5))
    return torch.randint(0, self.num_classes, (x.shape[0],))

  def load(self) -> None:
    """
      Not Implemented.

      Raises:
      ------
        NotImplementedError
    """
    raise NotImplementedError("A dummy is not expected to be used production")

  def parameters(self) -> Iterator[Parameter]:
    """
      Get the parameters of the neural network.

      Returns:
      --------
        Iterator[Parameter]
    """
    s = torch.nn.Sequential(torch.nn.Linear(10, self.num_classes))
    return s.parameters()

  def save(self) -> None:
    """
      Not Implemented.

      Raises:
      ------
        NotImplementedError
    """
    raise NotImplementedError("A dummy is not expected to be used production")

  def create(self) -> None:
    """
      Not Implemented.

      Raises:
      ------
        NotImplementedError
    """
    raise NotImplementedError("A dummy is not expected to be used production")

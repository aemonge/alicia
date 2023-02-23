import pytest, torch
from src.libs.shape_transforms import Reshapetransform, UnShapetransform

@pytest.fixture
def tensor_fixture():
  return torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

@pytest.fixture
def tensor_224_fixture():
  # Create a torch tensor with (1,28, 8) shape
  return torch.randn(1, 28, 8)

class Test_shape_transforms:
  def should_assing_self_to_shape_transforms(self):
    a = Reshapetransform((0,1))
    b = Reshapetransform((1,))
    assert a.shape == (0,1)
    assert b.shape == (1,)

  def should_return_self_if_shape_is_not_equal(self):
    a = Reshapetransform((0,1))
    b = Reshapetransform((1,2))
    assert a.shape != (1,2)
    assert b.shape != (0,1)

  def should_reshape_to_1D_the_tensor(self, tensor_fixture):
    a = Reshapetransform((-1,))
    shape = a(tensor_fixture).shape
    assert shape == torch.Size([12])

  def should_reshape_to_2D_the_tensor(self, tensor_fixture):
    # A 2D tensor of size 12 can be represented as a matrix of size 4x3.
    a = Reshapetransform((4,3,))
    shape = a(tensor_fixture).shape
    assert shape == torch.Size([4, 3])

  def should_reshape_to_3D_the_tensor(self, tensor_fixture):
    a = Reshapetransform((1, 4,3))
    shape = a(tensor_fixture).shape
    assert shape == torch.Size([1, 4, 3])

class TestUnShapetransform:
  # TODO: Remove when I remove this UnShapetransform
  def should_assing_self_to_shape_transforms(self):
    a = UnShapetransform((0,1))
    b = UnShapetransform((1,))
    assert a.shape == (0,1)
    assert b.shape == (1,)

  def should_shape_from_default_value(self, tensor_224_fixture):
    a = UnShapetransform((0,1))
    shape = a(tensor_224_fixture).shape
    assert shape == torch.Size([1, 28, 8])

  def should_careless_about_init_shape(self, tensor_fixture):
    a = UnShapetransform((0,1))
    shape = a(tensor_fixture, (1, 4,3)).shape
    assert shape == torch.Size([1, 4, 3])

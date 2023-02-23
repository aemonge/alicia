import torch
from src.libs.image_transforms import ImageToMatPlotLib

class TestImageToMatPlotLib:
  def should_init_with_shape(self):
    i = ImageToMatPlotLib((0,1))
    assert i.shape == (0,1)

  def should_transform_a_tensor_to_a_matplotlib_image(self):
    i = ImageToMatPlotLib((0,1))
    img = i(torch.rand(3,224,224))
    assert img.shape == (224,224,3)

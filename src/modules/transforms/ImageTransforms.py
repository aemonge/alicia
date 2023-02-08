import torch
from torchvision import transforms as Transforms

class UnShapeTransform(object):
  def __init__(self, shape):
    self.shape = shape

  def __call__(self, tensor):
    return torch.reshape(tensor, (1, 28, 28))

class ImageToMatPlotLib(object):
  """
    PyTorch tensors assume the color channel is the first dimension
      but matplotlib assumes is the third dimension
  """
  def __init__(self, shape):
    self.shape = shape

  def __call__(self, tensor):
    return tensor.numpy().transpose((1, 2, 0))

class ReshapeTransform(object):
  def __init__(self, shape):
    self.shape = shape

  def __call__(self, tensor):
    return torch.reshape(tensor, self.shape)
    # return image.view(image.shape[0], -1)

transforms = {
  "valid": Transforms.Compose([
    Transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
    Transforms.Resize(28),
    Transforms.CenterCrop(28),
    Transforms.ToTensor(),
    Transforms.Normalize((0.5,), (0.5,)),
    # ReshapeTransform((-1, )),
    # Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]),
  "display": Transforms.Compose([
    Transforms.Grayscale(),
    Transforms.ToTensor(),
    Transforms.Normalize((0.5,), (0.5,)),
    UnShapeTransform((-1, )),
    ImageToMatPlotLib(-1, ),
  ]),
  "test": Transforms.Compose([
    Transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
    Transforms.Resize(28),
    Transforms.CenterCrop(28),
    Transforms.ToTensor(),
    Transforms.Normalize((0.5,), (0.5,)),
    # ReshapeTransform((-1, )),
    # Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]),
  "train": Transforms.Compose([
    Transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
    Transforms.Resize(28),
    Transforms.CenterCrop(28),
    Transforms.ToTensor(),
    Transforms.Normalize((0.5,), (0.5,)),
    # ReshapeTransform((-1, )),
    # Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
}

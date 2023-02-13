from dependencies.core import torch, transforms

class UnShapetransform(object):
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

class Reshapetransform(object):
  def __init__(self, shape):
    self.shape = shape

  def __call__(self, tensor):
    return torch.reshape(tensor, self.shape)
    # return image.view(image.shape[0], -1)

ImageTransforms = {
  "valid": transforms.Compose([
    transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),
    Reshapetransform((-1, )),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]),
  "display": transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    UnShapetransform((-1, )),
    ImageToMatPlotLib(-1, ),
  ]),
  "test": transforms.Compose([
    transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    Reshapetransform((-1, )),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]),
  "train": transforms.Compose([
    transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    Reshapetransform((-1, )),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
}

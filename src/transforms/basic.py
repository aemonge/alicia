import torch
from torchvision import transforms as Transforms

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
    ReshapeTransform((-1, )),
    # Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]),
  "train": Transforms.Compose([
    Transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
    Transforms.Resize(28),
    Transforms.CenterCrop(28),
    Transforms.ToTensor(),
    Transforms.Normalize((0.5,), (0.5,)),
    ReshapeTransform((-1, )),
    # Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
}

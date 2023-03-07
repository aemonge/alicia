from dependencies.core import torchvision
from libs import ImageToMatPlotLib, UnShapetransform, Reshapetransform

def mnist_4D():
  return {
    "valid": torchvision.transforms.Compose([
      torchvision.transforms.Resize(28),
      torchvision.transforms.CenterCrop(28),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]),
    "display": torchvision.transforms.Compose([
      torchvision.transforms.Resize(28),
      torchvision.transforms.CenterCrop(28),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (0.5,)),
      ImageToMatPlotLib((-1, )),
    ]),
    "test": torchvision.transforms.Compose([
      torchvision.transforms.Grayscale(num_output_channels=3),
      torchvision.transforms.Resize(28),
      torchvision.transforms.CenterCrop(28),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]),
    "train": torchvision.transforms.Compose([
      torchvision.transforms.Resize(28),
      torchvision.transforms.CenterCrop(28),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (0.5,)),
    ])
  }

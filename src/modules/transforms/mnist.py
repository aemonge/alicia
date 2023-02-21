from dependencies.core import torchvision
from libs import ImageToMatPlotLib, UnShapetransform, Reshapetransform

def mnist():
  return {
    "valid": torchvision.transforms.Compose([
      torchvision.transforms.Grayscale(),
      torchvision.transforms.Resize(28),
      torchvision.transforms.CenterCrop(28),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (0.5,)),
      Reshapetransform((-1, )),
    ]),
    "display": torchvision.transforms.Compose([
      torchvision.transforms.Grayscale(),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (0.5,)),
      UnShapetransform((-1, )),
      ImageToMatPlotLib((-1, )),
    ]),
    "test": torchvision.transforms.Compose([
      torchvision.transforms.Grayscale(),
      torchvision.transforms.Resize(28),
      torchvision.transforms.CenterCrop(28),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (0.5,)),
      Reshapetransform((-1, )),
    ]),
    "train": torchvision.transforms.Compose([
      torchvision.transforms.Grayscale(),
      torchvision.transforms.Resize(28),
      torchvision.transforms.CenterCrop(28),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (0.5,)),
      Reshapetransform((-1, )),
    ])
  }

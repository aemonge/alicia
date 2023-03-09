from dependencies.core import torchvision
from libs import ImageToMatPlotLib, PadToSize

def img_pico():
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

def img_pico_czy():
  size = [28, 28]
  scale_factor = 2.2
  size_factored = [ round(x * scale_factor) for x in size ]
  return {
    "train": torchvision.transforms.Compose([
      PadToSize(size_factored),
      torchvision.transforms.CenterCrop(size_factored),
      torchvision.transforms.Resize(size_factored),

      torchvision.transforms.RandomAffine(42),
      torchvision.transforms.RandomInvert(),
      torchvision.transforms.RandomAdjustSharpness(2),
      torchvision.transforms.RandomAutocontrast(),
      torchvision.transforms.RandomPerspective(),

      torchvision.transforms.CenterCrop(size),
      torchvision.transforms.Resize(size),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]),
    "valid": torchvision.transforms.Compose([
      PadToSize(size_factored),
      torchvision.transforms.CenterCrop(size_factored),
      torchvision.transforms.Resize(size_factored),

      torchvision.transforms.RandomAdjustSharpness(2),
      torchvision.transforms.RandomAutocontrast(),

      torchvision.transforms.CenterCrop(size),
      torchvision.transforms.Resize(size),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]),
    "test": torchvision.transforms.Compose([
      torchvision.transforms.CenterCrop(size),
      torchvision.transforms.Resize(size),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]),
    "display": torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (0.5,)),
      ImageToMatPlotLib((-1, )),
    ]),
  }

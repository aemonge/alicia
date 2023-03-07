from dependencies.core import torchvision
from libs import ImageToMatPlotLib, PadToSize

def img_small():
  size = [320, 240]
  scale_factor = 2.2
  return {
    "valid": torchvision.transforms.Compose([
      PadToSize([ round(x * scale_factor) for x in size ]),
      torchvision.transforms.CenterCrop([ round(x * scale_factor) for x in size ]),
      torchvision.transforms.Resize(size),

      torchvision.transforms.RandomRotation(21),
      torchvision.transforms.RandomPerspective(),

      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "display": torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ImageToMatPlotLib((-1, )),
    ]),
    "test": torchvision.transforms.Compose([
      PadToSize([ round(x * scale_factor) for x in size ]),
      torchvision.transforms.CenterCrop([ round(x * scale_factor) for x in size ]),
      torchvision.transforms.Resize(size),

      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "train": torchvision.transforms.Compose([
      PadToSize([ round(x * scale_factor) for x in size ]),
      torchvision.transforms.CenterCrop([ round(x * scale_factor) for x in size ]),
      torchvision.transforms.Resize(size),

      torchvision.transforms.ColorJitter(),
      torchvision.transforms.RandomAffine(42),
      torchvision.transforms.RandomHorizontalFlip(),
      torchvision.transforms.RandomRotation(32),
      torchvision.transforms.RandomVerticalFlip(),
      torchvision.transforms.RandomRotation(21),
      torchvision.transforms.RandomPerspective(),

      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
  }

from dependencies.core import torchvision
from libs import ImageToMatPlotLib, PadToSize

def img_small_czy():
  size = [320, 240]
  scale_factors = [2, 2.2, 2,8]
  return {
    "train": torchvision.transforms.Compose([
      PadToSize([ round(x * scale_factors[1]) for x in size ]),
      torchvision.transforms.CenterCrop([ round(x * scale_factors[1]) for x in size ]),

      torchvision.transforms.ColorJitter(),
      torchvision.transforms.RandomAffine(42),
      torchvision.transforms.RandomHorizontalFlip(),
      torchvision.transforms.RandomRotation(32),
      torchvision.transforms.RandomVerticalFlip(),
      torchvision.transforms.RandomRotation(21),
      torchvision.transforms.RandomPerspective(),

      torchvision.transforms.Resize(size),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "valid": torchvision.transforms.Compose([
      PadToSize([ round(x * scale_factors[2]) for x in size ]),
      torchvision.transforms.CenterCrop([ round(x * scale_factors[2]) for x in size ]),

      torchvision.transforms.RandomRotation(21),
      torchvision.transforms.RandomPerspective(),

      torchvision.transforms.Resize(size),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "test": torchvision.transforms.Compose([
      PadToSize([ round(x * scale_factors[0]) for x in size ]),
      torchvision.transforms.CenterCrop([ round(x * scale_factors[0]) for x in size ]),

      torchvision.transforms.Resize(size),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "display": torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ImageToMatPlotLib((-1, )),
    ]),
  }

def img_small():
  size = [320, 240]
  scale_factors = [2, 2.2, 2,8]
  return {
    "train": torchvision.transforms.Compose([
      PadToSize([ round(x * scale_factors[1]) for x in size ]),
      torchvision.transforms.CenterCrop([ round(x * scale_factors[1]) for x in size ]),
      torchvision.transforms.Resize(size),

      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "valid": torchvision.transforms.Compose([
      PadToSize([ round(x * scale_factors[2]) for x in size ]),
      torchvision.transforms.CenterCrop([ round(x * scale_factors[2]) for x in size ]),
      torchvision.transforms.Resize(size),

      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "test": torchvision.transforms.Compose([
      PadToSize([ round(x * scale_factors[0]) for x in size ]),
      torchvision.transforms.CenterCrop([ round(x * scale_factors[0]) for x in size ]),
      torchvision.transforms.Resize(size),

      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "display": torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ImageToMatPlotLib((-1, )),
    ]),
  }

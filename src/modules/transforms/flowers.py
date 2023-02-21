from dependencies.core import torchvision
from libs import ImageToMatPlotLib, UnShapetransform, Reshapetransform

def flowers():
  return {
    "valid": torchvision.transforms.Compose([
      torchvision.transforms.RandomRotation(21),
      torchvision.transforms.RandomPerspective(),
      torchvision.transforms.Resize(228),
      torchvision.transforms.CenterCrop(228),

      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      Reshapetransform((-1, )),
    ]),
    "display": torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      UnShapetransform((-1, )),
      ImageToMatPlotLib((-1, )),
    ]),
    "test": torchvision.transforms.Compose([
      torchvision.transforms.Resize(228),
      torchvision.transforms.CenterCrop(228),

      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      Reshapetransform((-1, )),
    ]),
    "train": torchvision.transforms.Compose([
      torchvision.transforms.ColorJitter(),
      torchvision.transforms.RandomAffine(42),
      torchvision.transforms.RandomHorizontalFlip(),
      torchvision.transforms.RandomRotation(32),
      torchvision.transforms.RandomVerticalFlip(),
      torchvision.transforms.RandomRotation(21),
      torchvision.transforms.RandomPerspective(),
      torchvision.transforms.Resize(228),
      torchvision.transforms.CenterCrop(228),

      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      Reshapetransform((-1, )),
    ])
  }

from dependencies.core import torchvision
from libs import ImageToMatPlotLib, UnShapetransform, Reshapetransform

Flowers_Transforms = {
  "valid": torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]),
  "display": torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]),
  "test": torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]),
  "train": torchvision.transforms.Compose([
    torchvision.transforms.ColorJitter(),
    torchvision.transforms.RandomAffine(42),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(32),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomRotation(21),
    torchvision.transforms.RandomPerspective(),

    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
}

MNIST_28_Transforms = {
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

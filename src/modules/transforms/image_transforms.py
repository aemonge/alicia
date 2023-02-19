from dependencies.core import torchvision
from libs import ImageToMatPlotLib, UnShapetransform, Reshapetransform

ImageTransforms = {
  "valid": torchvision.transforms.Compose([
    # torchvision.transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
    torchvision.transforms.Resize(28),
    torchvision.transforms.CenterCrop(28),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.5,), (0.5,)),
    # Reshapetransform((-1, )),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]),
  "display": torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,)),
    UnShapetransform((-1, )),
    ImageToMatPlotLib((-1, )),
  ]),
  "test": torchvision.transforms.Compose([
    # torchvision.transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
    torchvision.transforms.Resize(28),
    torchvision.transforms.CenterCrop(28),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,)),
    # Reshapetransform((-1, )),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]),
  "train": torchvision.transforms.Compose([
    # torchvision.transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
    torchvision.transforms.Resize(28),
    torchvision.transforms.CenterCrop(28),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,)),
    # Reshapetransform((-1, )),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
}

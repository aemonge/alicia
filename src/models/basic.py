# pylint: disable=missing-module-docstring
import time
import os as Os
import numpy as Np
import plotext as Plt
from matplotlib import pyplot as Pyplot
import torch as Torch

from torchvision import transforms as Transforms
from torch.utils.data import DataLoader
from torch import optim as Optim
from torch import nn as Nn

from datasets.ae_image_dataset import AeImageDataset
# pylint: enable=missing-module-docstring

BATCH_SIZE = 4

class BasicModel:
  """
    A Basic Model aiming to guess the numbers from MNIST dataset.

    Attributes
    ----------
      verbose : boolean
        Whether to print the model information.
      print : function
        Function to print messages.
      data_dir : string
        The path to the MNIST dataset folder.
      epochs : integer
        The number of epochs to train the model.

    Constants
    ----------
      TRAIN_DATA_SIZE : float
        The size of the training dataset.
      TEST_DATA_SIZE : float
        The size of the test dataset.
      VALIDATION_DATA_SIZE : float
        The size of the validation dataset.
      IMG_SIZE : integer
        The size of the images.

    Methods
    -------
  """

  TRAIN_DATA_SIZE = .7
  TEST_DATA_SIZE = 0.2
  VALIDATION_DATA_SIZE = 0.1
  IMG_SIZE = 28

  def __init__(self, data_dir=None, step_print_fn=None, epochs=3, verbose=False):
    """
      Constructor.

      Parameters
      ----------
        data_dir : string
          The path to the MNIST dataset folder.
        step_print_fn : function
          Function to print messages.
        epochs : integer
          The number of epochs to train the model.
        verbose : boolean
          Whether to print the model information.
    """
    self.verbose = verbose
    self.print = step_print_fn
    self.data_dir = data_dir
    self.epochs = epochs

    self.__data = { "train": [], "test": [], "validation": [] }
    self.__criterion = Nn.CrossEntropyLoss()

    self.__model = Nn.Sequential(
      Nn.Linear(self.IMG_SIZE * self.IMG_SIZE, 128),
      Nn.ReLU(),
      Nn.Linear(128, 64),
      Nn.ReLU(),
      Nn.Linear(64, 10),
      Nn.LogSoftmax(dim=1)
    )
    self.__transform = Transforms.Compose([
      Transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
      Transforms.ToTensor(),
      Transforms.Normalize((0.5,), (0.5,))
    ])
    self.__optimizer = Optim.SGD(self.__model.parameters(), lr=0.003, momentum=0.9)
    self.__dataset = AeImageDataset(self.data_dir, transform = self.__transform)
    self.__loaders = {
      "train": DataLoader(self.__dataset, batch_size = BATCH_SIZE, shuffle=True),
    }
    self.__analytics = {
      "training": {
        "loss": 0 # mean, std
      },
    }

    if self.verbose:
      images, labels = next(iter(self.__loaders['train']))
      verbose_info = {
        'images': images,
        'labels': labels,
        'model': self.__model,
        'classes': self.__dataset.class_map
      }
      self.print.header(model="Basic", verbose = verbose_info)
    else:
      self.print.header(model="Basic")

  def train(self):
    """
      Run the model.
    """
    for epoch in range(self.epochs):
      self.__analytics['training']['loss'] = 0

      for (images, labels) in iter(self.__loaders['train']): # Id -> Label
        self.__optimizer.zero_grad()

        # Let's reshape as we just learned by experimenting 🤟
        images = images.view(images.shape[0], -1)
        labels = labels.long() # [:,0].long()
        output = self.__model(images)
        loss = self.__criterion(output, labels)

        loss.backward()
        self.__optimizer.step()

        self.__analytics['training']['loss'] += loss.item()
      else:
        training_loss = self.__analytics['training']['loss'] / len(self.__loaders['train'])
        print(f"  Epoch: {epoch}\t\t Training loss: {round(training_loss, 6)}")

    self.print.footer()

  def preview(self, image_count = 1):
    """
      Preview the model results by selection random images
    """

    for _ in range(image_count):
      images, labels = next(iter(self.__loaders['train']))
      img = images[0].view(1, 784)
      # Turn off gradients to speed up this part
      with Torch.no_grad():
        logps = self.__model(img)

      self.print.test(self.print_resutls, img, Torch.exp(logps))

  def print_resutls(self, img, probabilities_classes):
    """
      Print the results of the model.

      Parameters
      ----------
        img : torch.Tensor
          The input image.
        probabilities_classes : torch.Tensor
          The output of the network.
    """
    self._plot_wrapper()
    self._plot_img(img)
    self._plot_probabilities_classes(probabilities_classes)
    Plt.show()
    Plt.clf()

  def splitData(self):
    """
      Read directory and store all the file names in a list.
    """
    # TODO: Make the split to be random.
    file_list = []
    for path, subdirs, files in Os.walk(self.data_dir): # pylint: disable=unused-variable
      for name in files:
        file_no_extension = Os.path.splitext(name)[0]
        file_list.append({"file": Os.path.join(name), "label": file_no_extension})

    from_id = 0
    to_id = int(len(file_list) * self.TRAIN_DATA_SIZE)
    self.__data['train'] = file_list[from_id:to_id]

    from_id = to_id
    to_id += int(len(file_list) * self.TEST_DATA_SIZE)

    self.__data['test'] = file_list[from_id:to_id]
    from_id = to_id
    to_id += int(len(file_list) * self.VALIDATION_DATA_SIZE)

    self.__data['validation'] = file_list[from_id:to_id]

  def __tensor_to_image(self, img, img_path):
    """
      Convert a tensor to an image and saved in the img_path.

      Parameters
      ----------
        img : torch.Tensor
          The tensor to convert.
        img_path : string
          The path to the image.
    """
    img = img.view(1, 28, 28).permute(1, 2, 0)
    Pyplot.imshow(img)
    Pyplot.axis('off')
    Pyplot.savefig(img_path, bbox_inches='tight', pad_inches=0, transparent=True)

  def _plot_wrapper(self):
    """
      Plot the results of the test in a nice box with class map and the image.
    """
    Plt.subplots(1, 2)
    Plt.plot_size(73, 20)

  def _plot_img(self, img):
    """
      Plot the image in it's right section

      Parameters
      ----------
        img : torch.Tensor
          The tensor image to display.
    """
    tmp_path = 'tmp/mnist-number.jpg'
    self.__tensor_to_image(img, tmp_path)
    Plt.subplot(1, 2)
    Plt.image_plot(tmp_path)

  def _plot_probabilities_classes(self, probabilities_classes):
    """
      Plot class map in it's left section

      Parameters
      ----------
        probabilities_classes : torch.Tensor
          The tensor probabilities_classes to display.
    """
    classes = list(self.__dataset.class_map.keys())
    probabilities_classes = Np.array(probabilities_classes.view(10))
    probabilities_classes = Np.round(probabilities_classes, 2)
    Plt.subplot(1, 1)
    Plt.bar(classes, probabilities_classes, width= 0.1, orientation='h')

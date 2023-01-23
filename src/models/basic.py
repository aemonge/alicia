# pylint: disable=missing-module-docstring
import time
import os as Os
import numpy as Np
import plotext as Plt
from matplotlib import pyplot as Pyplot
import torch as Torch
import numpy as Numpy

from torchvision import transforms as Transforms
from torch.utils.data import DataLoader
from torch import optim as Optim
from torch import nn as Nn

from datasets.ae_image_dataset import AeImageDataset
from lib.dispaly_analytics import DispalyAnalytics as print_da
# pylint: enable=missing-module-docstring

BATCH_SIZE = 16

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

  def __init__(self, data_dir=None, step_print_fn=print_da(), epochs=9, verbose=False):
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

    self.__criterion = Nn.CrossEntropyLoss()

    self.__model = Nn.Sequential(
      Nn.Linear(self.IMG_SIZE * self.IMG_SIZE, 128),
      Nn.ReLU(),
      Nn.Linear(128, 64),
      Nn.ReLU(),
      Nn.Linear(64, 10),
      Nn.LogSoftmax(dim=1)
    )
    train_transform = Transforms.Compose([
      Transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
      Transforms.ToTensor(),
      Transforms.Normalize((0.5,), (0.5,))
    ])
    test_transform = Transforms.Compose([
      Transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
      Transforms.ToTensor(),
      Transforms.Normalize((0.5,), (0.5,))
    ])
    self.__optimizer = Optim.SGD(self.__model.parameters(), lr=0.003, momentum=0.9)
    self.__train_dataset = AeImageDataset(f"{self.data_dir}/train", transform = train_transform)
    self.__test_dataset = AeImageDataset(f"{self.data_dir}/test", transform = test_transform)
    self.__loaders = {
      "train": DataLoader(self.__train_dataset, batch_size = BATCH_SIZE, shuffle=True),
      "test": DataLoader(self.__test_dataset, batch_size = BATCH_SIZE, shuffle=True),
    }
    self.__analytics = {
      "training": {
        "loss": 0 # mean, std
      },
      "test": {
        "loss": 0 # mean, std
      },
    }

    if self.verbose:
      images, labels = next(iter(self.__loaders['train']))
      verbose_info = {
        'images': images,
        'labels': labels[0],
        'model': self.__model,
        'classes': self.__train_dataset.class_map
      }
      self.print.header(model="Basic", verbose = verbose_info)
    else:
      self.print.header(model="Basic")

  def train(self):
    """
      Run the model.
    """
    test_loader_count = len(self.__loaders['test'].dataset)
    train_loader_count = len(self.__loaders['train'].dataset)
    train_losses, test_losses = [], []
    time_count = 0.0
    start_time = time.time()

    for epoch in range(self.epochs):
      self.__analytics['training']['loss'] = 0

      for (images, labels) in iter(self.__loaders['train']): # Id -> Label
        self.__optimizer.zero_grad()

        # Let's reshape as we just learned by experimenting 🤟
        images = images.view(images.shape[0], -1)
        labels = labels[0].long() # [:,0].long()
        output = self.__model(images)
        loss = self.__criterion(output, labels)

        loss.backward()
        self.__optimizer.step()

        self.__analytics['training']['loss'] += loss.item()
      else:
        self.__analytics['test']['loss'] = 0
        test_correct = 0

        with Torch.no_grad(): # When validating, make it fast so no grad ;)
          self.__model.eval()
          for (images, labels) in iter(self.__loaders['test']): # Id -> Label
            images = images.view(images.shape[0], -1)
            labels = labels[0].long() # [:,0].long()
            log_ps = self.__model(images)
            loss = self.__criterion(log_ps, labels)
            self.__analytics['test']['loss'] += loss.item()

            ps = Torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_correct += equals.sum().item()

        self.__model.train()
        training_loss = self.__analytics['training']['loss'] / train_loader_count
        test_loss = self.__analytics['test']['loss'] / test_loader_count

        # At completion of epoch
        train_losses.append(self.__analytics['training']['loss'])
        test_losses.append(self.__analytics['test']['loss'])

        time_now = time.time()
        time_count += time_now - start_time
        print(
          f" Epoch: {epoch+1}/{self.epochs}, ",
          f" Time: {(time_now - start_time):.4f}s, "
          f" Accuracy: {(test_correct * 100 / test_loader_count):.2f}%"
        )
        print( f"   Loss: [ Train: {training_loss:.4f}, Test: {test_loss:.4f} ]")
        start_time = time_now

    self.__plot_loss(train_losses, test_losses)
    self.print.footer(total_time=time_count, accuracy=(test_correct * 100 / test_loader_count))

  def preview(self, image_count = 1):
    """
      Preview the model results by selection random images
    """

    for _ in range(image_count):
      images, _ = next(iter(self.__loaders['train']))
      img = images[0].view(1, 784)
      # Turn off gradients to speed up this part
      with Torch.no_grad():
        logps = self.__model(img)

      self.print.test(self.print_resutls, img, Torch.exp(logps))

  def __get_guessed_class(self, logps):
    listed_class = list(self.__train_dataset.class_map.keys())
    logps = Torch.exp(logps)
    logps = Np.array(logps.view(10))
    logps = Np.round(logps, 2)

    guess_id = Numpy.argmax(logps)
    return listed_class[guess_id]

  def call(self, output_directory = 'out'):
    class_keys = self.__train_dataset.class_map.keys()
    transform =  Transforms.Compose([
      Transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
      Transforms.ToTensor(),
      Transforms.Normalize((0.5,), (0.5,))
    ])
    self.__model.eval()
    data = AeImageDataset(
      output_directory, class_map=set(class_keys), transform=transform
    )
    loader = DataLoader(data, batch_size = BATCH_SIZE)
    csv = []

    for (images, labels) in iter(loader):
      for ix, img in enumerate(images):
        img = img.view(1, 784)
        with Torch.no_grad():
          logps = self.__model(img)

        csv.append(f"{ labels[1][ix] },{ self.__get_guessed_class(logps) }")
        # self.print.test(self.print_resutls, img, Torch.exp(logps))
    return csv

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
    Plt.clf()
    self.__plot_wrapper()
    self.__plot_img(img)
    self.__plot_probabilities_classes(probabilities_classes)
    Plt.show()
    Plt.clf()

  def __plot_loss(self, train_losses, test_losses):
    Plt.plot(train_losses, label="train")
    Plt.plot(test_losses, label="test")
    Plt.plot_size(73, 20)
    Plt.show()

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

  def __plot_wrapper(self):
    """
      Plot the results of the test in a nice box with class map and the image.
    """
    Plt.subplots(1, 2)
    Plt.plot_size(73, 20)

  def __plot_img(self, img):
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

  def __plot_probabilities_classes(self, probabilities_classes):
    """
      Plot class map in it's left section

      Parameters
      ----------
        probabilities_classes : torch.Tensor
          The tensor probabilities_classes to display.
    """
    classes = list(self.__train_dataset.class_map.keys())
    probabilities_classes = Np.array(probabilities_classes.view(10))
    probabilities_classes = Np.round(probabilities_classes, 2)
    Plt.subplot(1, 1)
    Plt.bar(classes, probabilities_classes, width= 0.1, orientation='h')

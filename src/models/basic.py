# pylint: disable=missing-module-docstring
import time
import numpy as Np
import plotext as Plt
from matplotlib import pyplot as Pyplot
import torch as Torch

from torchvision import transforms as Transforms
from torch.utils.data import DataLoader
from torch import optim as Optim
from torch import nn as Nn

from datasets.alicia_dataset import AliciaDataset
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

  def __init__(self, data_dir=None, step_print_fn=print_da(), epochs=1, verbose=False, model_file=None):
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
    self.class_map =  {
      "Ankle boot": 0, "Bag": 0, "Coat": 0, "Dress": 0, "Pullover": 0,
      "Sandal": 0, "Shirt": 0, "Sneaker": 0, "Top": 0, "Trouser": 0
    }
    self.__loaders = { "train": None, "test": None }
    self.__create_model()

    if model_file:
      self.__load_model(model_file)
    else:
      self.__init_model()

  def train(self):
    """
      Train the neural network model.

      Parameters
      ----------

      Returns
      -------
        None
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
            _, top_class = ps.topk(1, dim=1)
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

  def save(self, path):
    """
      Save the neural network model.

      Parameters
      ----------
        path : str
          The location where to save the neural network model.

      Returns
      -------
        None
    """
    Torch.save({'state_dict': self.__model.state_dict(), 'class_map': self.__train_dataset.class_map}, path)

  def preview(self, image_count = 1, path = ''):
    """
      Preview the model results by selection random images

      Parameters
      ----------
        image_count : integer
          The amount of images to display as a preview of the classification.
        path : string
          The location of the images.

      Returns
      -------
        None
    """

    dataset = AliciaDataset(path)
    loader = DataLoader(dataset, batch_size = 1, shuffle=True),

    for _ in range(image_count):
      images, _ = next(iter(loader))
      img = images[0].view(1, 784)
      # Turn off gradients to speed up this part
      with Torch.no_grad():
        logps = self.__model(img)

      self.print.test(self.print_resutls, img, Torch.exp(logps))

  def call(self, output_directory = 'out'):
    """
      Runs the model to classify images, with the best prediction gotten.

      Parameters
      ----------
        output_directory : str
          The path of the images to classify.

      Returns
      -------
        None
    """
    class_keys = self.class_map.keys()
    transform =  Transforms.Compose([
      Transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
      Transforms.ToTensor(),
      Transforms.Normalize((0.5,), (0.5,))
    ])
    self.__model.eval()
    data = AliciaDataset(
      output_directory, class_map=set(class_keys), transform=transform
    )
    loader = DataLoader(data, batch_size = BATCH_SIZE)
    csv = []

    for (images, labels) in iter(loader):
      for idx, img in enumerate(images):
        img = img.view(1, 784)
        with Torch.no_grad():
          logps = self.__model(img)
        csv.append(f"{ labels[1][idx] },{ self.__get_guessed_class(logps) }")
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

      Returns
      -------
        None
    """
    Plt.clf()
    self.__plot_wrapper()
    self.__plot_img(img)
    self.__plot_probabilities_classes(probabilities_classes)
    Plt.show()
    Plt.clf()

  def __get_guessed_class(self, logps):
    """
      Get the most probable class for a given logps.

      Parameters
      ----------
        logps : [float]
          The array of logarithm probabilities.

      Returns
      -------
        None
    """
    listed_class = list(self.class_map.keys())
    logps = Torch.exp(logps)
    logps = Np.array(logps.view(10))
    logps = Np.round(logps, 2)

    guess_id = Np.argmax(logps)
    return listed_class[guess_id]

  def __init_model(self):
    """
      Initialize the model, with the transformations, optimizer, criterion, and analytics metrics.

      Parameters
      ----------

      Returns
      -------
        None
    """
    self.__criterion = Nn.CrossEntropyLoss()

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
    self.__train_dataset = AliciaDataset(f"{self.data_dir}/train", transform = train_transform)
    self.__test_dataset = AliciaDataset(f"{self.data_dir}/test",
                                        transform = test_transform, class_map = self.__train_dataset.class_map
                                        )
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

  def __create_model(self): # , input_size, hidden_layers, output_size):
    """
      Create the neural network model.

      Parameters
      ----------
        None

      Returns
      -------
        None
    """
    self.__model = Nn.Sequential(
      Nn.Linear(self.IMG_SIZE * self.IMG_SIZE, 128),
      Nn.ReLU(),
      Nn.Linear(128, 64),
      Nn.ReLU(),
      Nn.Linear(64, 10),
      Nn.LogSoftmax(dim=1)
    )

  def __load_model(self, path):
    """
      Load the model from a path to retrain it or use it to classify.

      Parameters
      ----------
        path : str
          The path where to load the model.

      Returns
      -------
        None
    """
    data = Torch.load(path)
    self.class_map = data['class_map']
    self.__model.load_state_dict(data['state_dict'])
    self.__model.eval()

  def __plot_loss(self, train_losses, test_losses):
    """
      Print as plot the losses during training and testing

      Parameters
      ----------
        train_losses : list
          The training losses.
        tes_losses : list
          The test losses.

      Returns
      -------
        None
    """
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

      Returns
      -------
        None
    """
    img = img.view(1, 28, 28).permute(1, 2, 0)
    Pyplot.imshow(img)
    Pyplot.axis('off')
    Pyplot.savefig(img_path, bbox_inches='tight', pad_inches=0, transparent=True)

  def __plot_wrapper(self):
    """
      Plot the results of the test in a nice box with class map and the image.

      Parameters
      ----------
        None

      Returns
      -------
        None
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

      Returns
      -------
        None
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

      Returns
      -------
        None
    """
    classes = list(self.class_map.keys())
    probabilities_classes = Np.array(probabilities_classes.view(10))
    probabilities_classes = Np.round(probabilities_classes, 2)
    Plt.subplot(1, 1)
    Plt.bar(classes, probabilities_classes, width= 0.1, orientation='h')

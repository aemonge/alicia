import time # pylint: disable=missing-module-docstring
import os as OS
import torch as Torch

from torchvision import transforms as Transforms
from torch.utils.data import DataLoader, Dataset
from torch import optim as Optim
from torch import nn as Nn
from natsort import natsorted
from PIL import Image

BATCH_SIZE = 4

class CustomDataset(Dataset):
  def __init__(self, main_dir, transform = None, label_transform = None):
    all_imgs = OS.listdir(main_dir)

    self.main_dir = main_dir
    self.transform = transform
    self.label_transform = label_transform
    self.total_imgs = natsorted(all_imgs)

    self.class_map = dict({
      'one': 0, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
      'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    })

  def __len__(self):
    return len(self.total_imgs)

  def __getitem__(self, idx):
    """
    Notes:
      SEE: https://stackoverflow.com/questions/51911749/what-is-the-difference-between-torch-tensor-and-torch-tensor
    Args:
      idx (int): Index
    Returns:
      tuple: (image, target) where target is index of the target class.
    """
    img_loc = OS.path.join(self.main_dir, self.total_imgs[idx])
    image = Image.open(img_loc).convert("RGB")

    if self.transform:
      image = self.transform(image)

    if self.label_transform:
      label = self.label_transform('dog')
    else:
      class_id = self.class_map['ten']
      # class_id = Torch.Tensor(data=[class_id]*10)
      # class_id = Torch.Tensor(data=[[class_id]*10]*10)
      # class_id = Torch.Tensor(data=[[[class_id]*10]])
      class_id = Torch.Tensor([0]*BATCH_SIZE)

    return image, class_id

class BasicModel:
  """
    A Basic Model aiming to guess the numbers from MNIST dataset.
  """

  TRAIN_DATA_SIZE = .7
  TEST_DATA_SIZE = 0.2
  VALIDATION_DATA_SIZE = 0.1
  IMG_SIZE = 28

  def __init__(self, data_dir=None, step_print_fn=None, epochs=3, verbose=False):
    self.verbose = verbose
    self.print = step_print_fn
    self.data_dir = data_dir
    self.data = { "train": [], "test": [], "validation": [] }
    self.epochs = epochs
    # self.model = Nn.Sequential(
    #   Nn.Linear(self.START_SIZE, 10),
    #   Nn.LogSoftmax(dim=1), # 784
    # )

    # 4 * 28 = 112
    self.model = Nn.Sequential(
      # Nn.Linear(28, self.IMG_SIZE * self.IMG_SIZE), # I should NOT adjust this, I expect the in WxH input
      # Nn.ReLU(),
      Nn.Linear(self.IMG_SIZE * self.IMG_SIZE, 128),
      Nn.ReLU(),
      Nn.Linear(128, 64),
      Nn.ReLU(),
      Nn.Linear(64, 10),
      Nn.LogSoftmax(dim=1)
    )
    self.transform = Transforms.Compose([
      # Transforms.Resize(self.START_SIZE),
      Transforms.Grayscale(), # Changes the size to [1, 1, 28, 28] [batch, channels, width, height]
      Transforms.ToTensor(),
      Transforms.Normalize((0.5,), (0.5,))
      # , Transforms.Normalize(
      #   mean=[0.485, 0.456, 0.406],
      #   std=[0.229, 0.224, 0.225]
      # )
    ])
    self.criterion = Nn.NLLLoss()

    self.my_dataset = CustomDataset(self.data_dir, transform = self.transform)
    self.loaders = {
      "train": DataLoader(self.my_dataset, batch_size = BATCH_SIZE),
    }
    self.analytics = {
      "training": {
        # "mean": random.random(),
        # "std": random.randint(0, 5),
        "loss": 0
      },
    }

    if self.verbose:
      for id, (images, labels) in enumerate(self.loaders['train']):
        if id == 0: # TODO: There *has* to be a better way to get to this dat without looping
          self.print.header(model="Basic", verbose = { 'images': images, 'labels': labels, 'model': self.model })
    else:
      self.print.header(model="Basic")

  def run(self):
    timer = time.time()

    optimizer = Optim.SGD(self.model.parameters(), lr=0.003, momentum=0.9)
    criterion = Nn.CrossEntropyLoss()

    for epoch in range(self.epochs):
      running_loss = 0

      for (images, labels) in iter(self.loaders['train']): # Id -> Label
        optimizer.zero_grad()

        # Let's reshape as we just learned by experimenting 🤟
        images = images.view(images.shape[0], -1)
        labels = labels[:,0].long()
        output = self.model(images)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        self.analytics['training']['loss'] += loss.item()
      else:
        training_loss = self.analytics['training']['loss'] / len(self.loaders['train'])
        print(f"Training loss: {training_loss}")

    self.print.footer()

  def splitData(self):
    # TODO: Make the split to be random.
    """
      Read directory and store all the file names in a list.
    """
    file_list = []
    for path, subdirs, files in OS.walk(self.data_dir): # pylint: disable=unused-variable
      for name in files:
        file_no_extension = OS.path.splitext(name)[0]
        file_list.append({"file": OS.path.join(name), "label": file_no_extension})

    from_id = 0
    to_id = int(len(file_list) * self.TRAIN_DATA_SIZE)
    self.data['train'] = file_list[from_id:to_id]

    from_id = to_id
    to_id += int(len(file_list) * self.TEST_DATA_SIZE)

    self.data['test'] = file_list[from_id:to_id]
    from_id = to_id
    to_id += int(len(file_list) * self.VALIDATION_DATA_SIZE)

    self.data['validation'] = file_list[from_id:to_id]

"""
@SEE: [~/udacity/introduction-pAIthon-devs/neural-networks/deep-learning/student-admissions](http://localhost:8888/notebooks/StudentAdmissions.ipynb)

def train_nn(features, targets, epochs, learnrate):

    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here
            #   rather than storing h as a separate variable
            output = sigmoid(np.dot(x, weights))

            # The error, the target minus the network output
            error = error_formula(y, output)

            # The error term
            #   Notice we calulate f'(h) here instead of defining a separate
            #   sigmoid_prime function. This just makes it faster because we
            #   can re-use the result of the sigmoid function stored in
            #   the output variable
            error_term = error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times the inputs
            del_w += error_term * x

        # Update the weights here. The learning rate times the
        # change in weights, divided by the number of records to average
        weights += learnrate * del_w / n_records

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights
"""

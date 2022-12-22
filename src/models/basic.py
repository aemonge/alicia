import time # pylint: disable=missing-module-docstring
import os as OS
from torch import optim as Optim
from torch import nn as Nn
from PIL import Image
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as Transforms
import plotext as plt

class CustomDataset(Dataset):
  def __init__(self, main_dir, transform):
    self.main_dir = main_dir
    self.transform = transform
    all_imgs = OS.listdir(main_dir)
    self.total_imgs = natsorted(all_imgs)

  def __len__(self):
    return len(self.total_imgs)

  def __getitem__(self, idx):
    img_loc = OS.path.join(self.main_dir, self.total_imgs[idx])
    image = Image.open(img_loc).convert("RGB")
    tensor_image = self.transform(image)
    return tensor_image

class BasicModel:
  TRAIN_DATA_SIZE = .7
  TEST_DATA_SIZE = 0.2
  VALIDATION_DATA_SIZE = 0.1
  """
    A Basic Model aiming to guess the numbers from MNIST dataset.
  """
  def __init__(self, data_dir=None, step_print_fn=None, epochs=3):
    # if data is None:
    #   raise Exception("Please provide a valid `.mat` file")
    # else :
    #   self.mnist_numbers = loadmat(data)

    self.print = step_print_fn
    self.data_dir = data_dir
    self.data = { "train": [], "test": [], "validation": [] }
    self.epochs = epochs
    self.model = Nn.Sequential(
      Nn.Linear(28*28, 128), # 784
      Nn.ReLU(),
      Nn.Linear(128, 64),
      Nn.ReLU(),
      Nn.Linear(64, 10),
      Nn.LogSoftmax(dim=1)
    )
    self.transform = Transforms.Compose([
      # Transforms.Resize(255),
      # Transforms.CenterCrop(28),
      Transforms.ToTensor()
    ])
    self.criterion = Nn.NLLLoss()
    self.optimizer = Optim.SGD(self.model.parameters(), lr=0.003)
    # self.transform = Transforms.Compose([
    #   Transforms.ToTensor(),
    #   Transforms.Normalize((0.5,), (0.5,))
    # ])
    # datasets and loaders
    # raw_set = datasets.MNIST('data/mnist-numbers/raw', download=True)
    # train_set = datasets.MNIST('data/mnist-numbers/train', download=True, train=True, transform=self.transform)
    # validation_set = datasets.MNIST('data/mnist-numbers/validation', download=True, train=True, transform=self.transform)
    my_dataset = CustomDataset(self.data_dir, transform=self.transform)
    self.loaders = {
      "train": DataLoader(my_dataset, batch_size=32, shuffle=True),
      # "validation": DataLoader(validation_set, batch_size=64, shuffle=True)
    }

    # for idx, (img, _) in enumerate(raw_set): # NOTE: Only done once

    #   img.save('data/mnist-numbers/raw/{:05d}.jpg'.format(idx))

    # self.criterion = nn.CrossEntropyLoss()
    # self.optimizer = nn.optim.Adam(self.model.parameters(), lr=0,)
    self.analytics = {
      # "test": {
      #   "mean": random.random(),
      #   "std": random.randint(0, 5),
      #   "loss": random.randint(0, 10 - epoch)
      # },
      "training": {
        # "mean": random.random(),
        # "std": random.randint(0, 5),
        "loss": 0
      },
    }

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

  def train(self):
    """
      Train the model.
    """
    self.splitData()
    self.model.train()
    for epoch in range(self.epochs):
      for data in self.data['train']:
        self.optimizer.zero_

  def run(self):
    timer = time.time()
    self.print.header(model="Basic")

    # dataiter = iter(self.loaders["train"])
    # images, labels = next(dataiter)
    # print(images.shape)
    # print(labels.shape)
    # print(images[0])
    # plt.image_plot('data/mnist-numbers/00004.jpg', grayscale=True)
    # plt.plot_size(height=20, width=35)
    # plt.show()

    # print(self.mnist_numbers["data"].T)
    # print(self.mnist_numbers["label"][0])

    for epoch in range(self.epochs):
      # print(self.data['train'])
      running_loss = 0
      for x in self.loaders['train']:
        # file, label = x.values()
        # print(file, label)
        # self.optimizer.zero_grad()

        output = self.model(x)
        # loss = self.criterion(output, label)
        # loss.backward()
        # self.optimizer.step()
        # running_loss += loss.item()
        # print(file, label)
    else:
      print(f"Training loss: {running_loss/len(self.loaders['train'])}")

        # images = images.
    #   for images, labels in trianloader:

      # return
      # time.sleep(self.sleep_time)

      # analytics = {
      #   "test": {
      #     "mean": random.random(),
      #     "std": random.randint(0, 5),
      #     "loss": random.randint(0, 10 - epoch)
      #   },
      #   "training": {
      #     "mean": random.random(),
      #     "std": random.randint(0, 5),
      #     "loss": random.randint(0, 10 - epoch)
      #   },
      # }
      # self.steps["testAcc"].append(analytics["test"]["loss"])
      # self.steps["trainingAcc"].append(analytics["training"]["loss"])
      # self.print.step(epoch=epoch, run_time=(self.now() - timer), step_analysis=analytics)
      # timer = time.time()

    self.print.footer()


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

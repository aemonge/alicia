# 🐛 Tech debt

* [ ] Make pylint understand venv paths rightfully.
* [ ] Make pylint and pyright to understand the callback method class, or create a abstract class

# External documentation

> Useful links found and used while developing this

* /Users/aemonge/udacity/introduction-pAIthon-devs/neural-networks/deep-learning-v2-pytorch/intro-to-pytorch
* (https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d)
* (https://stackoverflow.com/questions/51911749/what-is-the-difference-between-torch-tensor-and-torch-tensor)

## Part 0 - Own Requirements to achieve the other parts

### With a dummy model

* [x] **CLI** Build the CLI arguments parser.
* [x] **CLI Analytics** display on the Analytics on an table on the CLI, to understand
  - Epoch number.
  - Learn per epoch.
  - Test Accuracy (mean) per epoch.
  - Test Accuracy (standard deviation std) per epoch.
  - Loss's (Training, Validation and Testing)
  - Time each epoch took to finish.
  - Was GPU enabled?.
  - Display the model information: hidden layer count and size of each hidden layer tensors.
* [x] **CLI Plot** Introduce a plot to compare the 'Training Loss' vs the 'Validation Loss' over epochs, to be able to
  see possible over-fitting of the algorithm chosen.
* [x] A class probability bar-graph to display the results form three random data picks.
* [ ] Semi automatic data splitter into: Training, Testing, and Verifying.

### Getting real with Number Recognition (mnist)
> (https://deepai.org/dataset/mnist)
> (https://medium.com/fenwicks/tutorial-1-mnist-the-hello-world-of-deep-learning-abd252c47709)

Re-do previous steps but now with a custom and extended real AI algorithms.

* [x] Implement a argument to choose different models to use.
* [-] Implement the most simplistic neural network with none sequential functions
      - Backtrace for training.
      - Disable backtrace for testing.
      - Disable backtrace for Verification.
* [ ] Implement a copy of the previous neural network using sequential function.
* [ ] Get extremely crazy and creative to create a super complex model.
* [ ] Test all three models and compare them, with you Analytics.
* [ ] Extend your Analytics, to be able to have a one-on-one comparison of the three algorithms.
* [ ] Create a sensible fourth algorithm, as a chosen algorithm.
      - Make sure this has the best results seen with your Analytics.
* [ ] Add a fifth algorithm by downloading and using a pre-trained model.
* [ ] Run such a model against your Analytics.
* [ ] Add a sixth or seventh downloaded pre-trained model.
* [ ] Choose one of such models, and extend (transfer learning) it as a custom one.

#### Model Names
1. [x] Dummy
2. [-] Basic
3. [ ] Crazy
4. [ ] Optimum
5. [ ] [downloaded name]
6. [ ] [downloaded name]
7. [ ] [downloaded name]
8. [ ] AeM-[downloaded name]

### Approaching Udacity
> (https://github.com/zalandoresearch/fashion-mnist)

* [ ] Extend all your implementation to Recognition of the [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
* [ ] Extend all your implementations to Recognition of the base Cat/Dogs/Breads data from the pre-project
* [ ] Extend all your implementation to Recognition of the data from this project.

## Part 1 - Development Notebook

* [ ] **Package Imports** All the necessary packages and modules are imported in the first cell of the notebook
* [ ] **Training data augmentation** torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping
* [ ] **Data normalization** The training, validation, and testing data is appropriately cropped and normalized
* [ ] **Data loading** The data for each set (train, validation, test) is loaded with torchvision's ImageFolder
* [ ] **Data batching** The data for each set is loaded with torchvision's DataLoader
* [ ] **Pretrained Network** A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen
* [ ] **Feedforward Classifier** A new feedforward network is defined for use as a classifier using the features as input
* [ ] **Training the network** The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static
* [ ] **Validation Loss and Accuracy** During training, the validation loss and accuracy are displayed
* [ ] **Testing Accuracy** The network's accuracy is measured on the test data
* [x] **Saving the model** The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary
* [ ] **Loading checkpoints** There is a function that successfully loads a checkpoint and rebuilds the model
* [ ] **Image Processing** The process_image function successfully converts a PIL image into an object that can be used as input to a trained model
* [ ] **Class Prediction** The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image
* [ ] **Sanity Checking with matplotlib** A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names

## Part 2 - Command Line Application

* [ ] **Training a network** train.py successfully trains a new network on a dataset of images
* [ ] **Training validation log** The training loss, validation loss, and validation accuracy are printed out as a network trains
* [ ] **Model architecture** The training script allows users to choose from at least two different architectures available from torchvision.models
* [ ] **Model hyperparameters** The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
* [ ] **Training with GPU** The training script allows users to choose training the model on a GPU
* [ ] **Predicting classes** The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
* [ ] **Top K classes** The predict.py script allows users to print out the top K classes along with associated probabilities
* [ ] **Displaying class names** The predict.py script allows users to load a JSON file that maps the class values to other category names
* [ ] **Predicting with GPU** The predict.py script allows users to use the GPU to calculate the predictions

## Part 3 - Personal Extra
> (https://deepai.org/datasets)

* [ ] Choose three data set form deep-ai.org and test with new data
* [ ] Create a image classifier for your dads works, and classify all his images of architecture


.. image:: https://github.com/aemonge/alicia/raw/main/docs/DallE-Alicia-logo.jpg
   :width: 75px
   :align: left

.. image:: https://img.shields.io/badge/badges-awesome-green.svg
   :target: https://github.com/Naereen/badges

.. image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/v/ansicolortags.svg
   :target: https://pypi.python.org/pypi/alicia/

.. image:: https://img.shields.io/pypi/dm/ansicolortags.svg
   :target: https://pypi.python.org/pypi/alicia/

.. image:: https://img.shields.io/pypi/l/ansicolortags.svg
   :target: https://pypi.python.org/pypi/ansicolortags/

.. image:: https://img.shields.io/badge/say-thanks-ff69b4.svg
   :target: https://saythanks.io/to/kennethreitz

================================================
                   AlicIA
================================================
::

  Usage: alicia [OPTIONS] COMMAND [ARGS]...

    A CLI to download, create, modify, train, test, predict and compare an image classifiers.

    Supporting mostly all torch-vision neural networks and datasets.

    This will also identify cute 🐱 or a fierce 🐶, also flowers or what type of
    🏘️ you should be.

  Options:
    -v, --verbose
    -g, --gpu
    --version      Show the version and exit.
    --help         Show this message and exit.

  Commands:
    compare   Compare the info, accuracy, and step speed two (or more by...
    create    Creates a new model for a given architecture.
    download  Download a MNIST dataset with PyTorch and split it into...
    info      Display information about a model architecture.
    modify    Changes the hyper parameters of a model.
    predict   Predict images using a pre trained model, for a given folder...
    test      Test a pre trained model.
    train     Train a given architecture with a data directory containing a...

View a FashionMNIST demo
-----------------------------------------------

.. image:: https://asciinema.org/a/561138.png
   :target: https://asciinema.org/a/561138?autoplay=1"

Install and usage
================================================
::

    pip install alicia
    alicia --help


If you just want to see a quick showcase of the tool, download and run `showcase.sh` https://github.com/aemonge/alicia/raw/main/docs/showcase.sh

Features
-----------------------------------------------

To see the full list of features, and option please refer to `alicia --help`

* Download common torchvision datasets (tested with the following):
    - MNIST
    - FashionMNIST
    - Flowers102
    - EMNIST
    - StanfordCars
    - KMNIST and CIFAR10
* Select different transforms to train.
* Train, test and predict using different custom-made and torch-vision models:
    - SqueezeNet
    - AlexNet
    - MNASNet
* Get information about each model.
* Compare models training speed, accuracy, and meta information.
* View test prediction results in the console, or with matplotlib.
* Adds the network training history log, to the model. To enhance the info and compare.
* Supports pre-trained models, with weights settings.
* Automatically set the input size based on the image resolution.

References
-----------------------------------------------

Useful links found and used while developing this

* https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d
* https://stackoverflow.com/questions/51911749/what-is-the-difference-between-torch-tensor-and-torch-tensor
* https://deepai.org/dataset/mnist
* https://medium.com/fenwicks/tutorial-1-mnist-the-hello-world-of-deep-learning-abd252c47709

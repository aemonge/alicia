
.. image:: https://img.shields.io/pypi/v/alicia.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/alicia/

.. image:: https://pepy.tech/badge/alicia/month
    :alt: Monthly Downloads
    :target: https://pepy.tech/project/alicia


================================================
                   AlicIA
================================================
::

  Usage: alicia [OPTIONS] COMMAND [ARGS]...

    A CLI to download, train, test, predict and compare an image classifiers.

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
    predict   Predict images using a pre trained model, for a given folder...
    test      Test a pre trained model.
    train     Train a given architecture with a data directory containing a...


.. image:: ./docs/DallE-Alicia-logo.png
    :alt: DallE-Alicia-logo

Install and usage
================================================
::

    pip install alicia
    alicia --help


If you just want to see a quick showcase of the tool, download and run `../showcase.sh`

Features
-----------------------------------------------

To see the full list of features, and option please refer to `alicia --help`

* Download common torchvision datasets
* Train, test and predict using different custom-made and torch-vision models.
* Get information about each model.
* Compare models training speed, accuracy, and meta information.
* Tested with MNIST and FashionMNIST.
* View results in the console, or with matplotlib

References
-----------------------------------------------

Useful links found and used while developing this

* (https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d)
* (https://stackoverflow.com/questions/51911749/what-is-the-difference-between-torch-tensor-and-torch-tensor)
* (https://deepai.org/dataset/mnist)
* (https://medium.com/fenwicks/tutorial-1-mnist-the-hello-world-of-deep-learning-abd252c47709)

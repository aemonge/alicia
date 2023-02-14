
.. image:: https://img.shields.io/pypi/v/aeimg-classifier.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/aeimg-classifier/

.. image:: https://img.shields.io/conda/vn/conda-forge/aeimg-classifier.svg
    :alt: Conda-Forge
    :target: https://anaconda.org/conda-forge/aeimg-classifier

.. image:: https://pepy.tech/badge/aeimg-classifier/month
    :alt: Monthly Downloads
    :target: https://pepy.tech/project/aeimg-classifier

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

================================================
                   AlicIA
================================================

.. image:: ./DallE-Alicia-logo.png

    A CLI to download, train, test, predict and compare an image classifiers.

    Supporting mostly all torch-vision neural networks and datasets.

    This will also identify cute 🐱 or a fierce 🐶, also flowers
    or what type of 🏘️ you should be.


To build locally and develop locally
================================================

Use tox builder and install the it with pip::

    tox .
    pip install -e .

Load tags into your editor::

    ctags -R

Use the test with pytest::

    pytest

TODO My list of pending and implemented features
================================================

! [ ] Bug 🐛: Fix the label tagging for the images, seams to be the root of evil.

1.  [ ] Implement the following CLI views:
        - [x] download
        - [x] train (same arguments as the udacity project)
        - [x] test (--with sanity check, console or visual)
        - [x] predict (same arguments as the udacity project)
        - [ ] compare (compare validation-accuracy, test-accuracy and time between archs)
2.  [x] Unit test, with performance testings, use fashion-mnist small data.
3.  [x] Simplify the files structure.
4.  [ ] Implement a behaviour to change hidden units on all models.
5.  [ ] Check with all of the following data:
        - MNIST Numbers
        - MNIST Fashion
        - Udacity Flowers
        - Cats and Dogs images
        - Cifar, EMNIST, CelebA, KMNIST, Omniglot, PhotoTour, Places265, SBU, STL10, USPS, VOC
6.  [ ] Deploy the project to pypy and include the basic documentation.
7.  [ ] Continue with the other models, on a TDD methodology.
        - [x] Dummy (with random sleep timers, used for testing)
        - [x] Basic (Simple nn.Sequential)
        - [ ] Raw (Create it with nn.Sequential, and implement the math fn)
        - [ ] Crazy (A really complex nn.Sequential)
        - [ ] SqueezeNet
8.  [ ] The abstract class should let you choose between architectures by name.
9.  [ ] Implement the AlexNet network, and start applying to companies.
10.  [ ] Complete AlicIA with all the models
        - [ ] VGG (AlexNet is fast; like how you would expect it)
        - [ ] Choose from deep-ai.org
        - [ ] Try to get a open-ai model.
        - [ ] Inception
        - [ ] DenseNet
        - [ ] Alicia (a mixed model, from my favorites)
11. [ ] Pay tech debt.

🐛 Tech debt (Know Issues)
================================================

* [ ] Make pylint understand venv paths rightfully.
* [ ] Make pylint and pyright to understand the callback method class, or create a abstract class
* [ ] Change `--verbose` for `--log-level [error, info, verbose]` and display none, less, all information.
* [ ] Make the creator dynamic, not hard-coded.
* [x] Define well structure architecture. Transforms (and anti-transforms) are bounded to the data.
      - As the classifiers
* [ ] Move both, the criterion and optimizer away from the model, and generate a split model for them.
      - While `alicia train` I should specify the learning-rate.
* [ ] Trim all the `tox` stuff that you aren't using.
* [ ] Re-enable the console pure view (`plottext`)
* [x] On the tests, make the lablel.csv smaller, only containing the actual labels
* [ ] Choose a default width, and adjust all output to such a width

References
================================================

Useful links found and used while developing this

* (https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d)
* (https://stackoverflow.com/questions/51911749/what-is-the-difference-between-torch-tensor-and-torch-tensor)
* (https://deepai.org/dataset/mnist)
* (https://medium.com/fenwicks/tutorial-1-mnist-the-hello-world-of-deep-learning-abd252c47709)

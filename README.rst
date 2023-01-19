.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/aeimg-classifier.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/aeimg-classifier
    .. image:: https://readthedocs.org/projects/aeimg-classifier/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://aeimg-classifier.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/aeimg-classifier/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/aeimg-classifier
    .. image:: https://img.shields.io/pypi/v/aeimg-classifier.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/aeimg-classifier/
    .. image:: https://img.shields.io/conda/vn/conda-forge/aeimg-classifier.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/aeimg-classifier
    .. image:: https://pepy.tech/badge/aeimg-classifier/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/aeimg-classifier
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/aeimg-classifier

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

================
aeimg-classifier
================


    Add a short description here!


A longer description of your project goes here...


.. _pyscaffold-notes:

Making Changes & Contributing
=============================

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd aeimg-classifier
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate

Don't forget to tell your contributors to also install and use pre-commit.

.. _pre-commit: https://pre-commit.com/

To build locally and develop locally

    tox .
    pip install -e .

Note
====

This project has been set up using PyScaffold 4.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

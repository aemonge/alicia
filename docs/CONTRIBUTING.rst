============
Contributing
============

Welcome to ``alicia`` contributor's guide.

This document focuses on getting any potential contributor familiarized
with the development processes, but `other kinds of contributions`_ are also
appreciated.

If you are new to using git_ or have never collaborated in a project previously,
please have a look at `contribution-guide.org`_. Other resources are also
listed in the excellent `guide created by FreeCodeCamp`_ [#contrib1]_.

Please notice, all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. When in doubt, `Python Software
Foundation's Code of Conduct`_ is a good reference in terms of behavior
guidelines.


Issue Reports
=============

If you experience bugs or general issues with ``alicia``, please have a look
on the `issue tracker`_. If you don't see anything useful there, please feel
free to fire an issue report.

.. tip::
   Please don't forget to include the closed issues in your search.
   Sometimes a solution was already reported, and the problem is considered
   **solved**.

New issue reports should include information about your programming environment
(e.g., operating system, Python version) and steps to reproduce the problem.
Please try also to simplify the reproduction steps to a very minimal example
that still illustrates the problem you are facing. By removing other factors,
you help us to identify the root cause of the issue.


Documentation Improvements
==========================

You can help improve ``alicia`` docs by making them more readable and coherent, or
by adding missing information and correcting mistakes.

``alicia`` documentation uses Sphinx_ as its main documentation compiler.
This means that the docs are kept in the same repository as the project code, and
that any documentation update is done in the same way was a code contribution.

Code Contributions
==================

Submit an issue
---------------

Before you work on any non-trivial code contribution it's best to first create
a report in the `issue tracker`_ to start a discussion on the subject.
This often provides additional considerations and avoids unnecessary work.

Create an environment
---------------------

Before you start coding, we recommend creating an isolated `virtual
environment`_ to avoid any problems with your installed Python packages.

anaconda_::

    conda create -n alicia
    conda activate alicia
    pip install poetry
    poetry install

Clone the repository
--------------------

Create an user account on |the repository service| if you do not already have one.
Fork the project repository_: click on the *Fork* button near the top of the
page. This creates a copy of the code under your account on |the repository service|.
Clone this copy to your local disk::

 git clone git@github.com:aemonge/alicia.git
 cd alicia

Install my chosen build tool::

    pip install poetry

Then, you should run to test locally::

   poetry install

Implement your changes
----------------------

Create a branch to hold your changes::

 git checkout -b my-feature

and start making changes. Never work on the main branch!

Start your work on this branch. Don't forget to add docstrings_ to new
functions, modules and classes, especially if they are part of public APIs.

Add yourself to the list of contributors in ``AUTHORS.rst``.

When you’re done editing, do::

 git add <MODIFIED FILES>
 git commit

to record your changes in git_.

.. todo:: if you are not using pre-commit, please remove the following item:

Please make sure to see the validation messages from |pre-commit|_ and fix
any eventual issues.
This should automatically use flake8_/black_ to check/fix the code style
in a way that is compatible with the project.

.. important:: Don't forget to add unit tests and documentation in case your
   contribution adds an additional feature and is not just a bugfix.

   Moreover, writing a `descriptive commit message`_ is highly recommended.
   In case of doubt, you can check the commit history with::

      git log --graph --decorate --pretty=oneline --abbrev-commit --all

   to look for recurring communication patterns.

Please check that your changes don't break any unit tests with::

    pytest

Submit your contribution
------------------------

If everything works fine, push your local branch to |the repository service| with::

 git push -u origin my-feature

Go to the web page of your fork and click |contribute button|
to send your changes for review.
::

      Find more detailed information in `creating a PR`_. You might also want to open
      the PR as a draft first and mark it as ready for review after the feedbacks
      from the continuous integration (CI) system or any required fixes.

References
================

.. _black: https://pypi.org/project/black/
.. _CommonMark: https://commonmark.org/
.. _contribution-guide.org: https://www.contribution-guide.org/
.. _creating a PR: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
.. _descriptive commit message: https://chris.beams.io/posts/git-commit
.. _docstrings: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _first-contributions tutorial: https://github.com/firstcontributions/first-contributions
.. _flake8: https://flake8.pycqa.org/en/stable/
.. _git: https://git-scm.com
.. _GitHub's fork and pull request workflow: https://guides.github.com/activities/forking/
.. _guide created by FreeCodeCamp: https://github.com/FreeCodeCamp/how-to-contribute-to-open-source
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _MyST: https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html
.. _other kinds of contributions: https://opensource.guide/how-to-contribute
.. _pre-commit: https://pre-commit.com/
.. _PyPI: https://pypi.org/
.. _PyScaffold's contributor's guide: https://pyscaffold.org/en/stable/contributing.html
.. _Pytest can drop you: https://docs.pytest.org/en/stable/how-to/failures.html#using-python-library-pdb-with-pytest
.. _Python Software Foundation's Code of Conduct: https://www.python.org/psf/conduct/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _tox: https://tox.wiki/en/stable/
.. _virtual environment: https://realpython.com/python-virtual-environments-a-primer/
.. _virtualenv: https://virtualenv.pypa.io/en/stable/

.. _GitHub web interface: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
.. _GitHub's code editor: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files

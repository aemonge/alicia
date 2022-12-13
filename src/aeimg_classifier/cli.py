"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = aeimg_classifier.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import sys
import click

from aeimg_classifier import __version__

__author__ = "aemonge"
__copyright__ = "aemonge"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from aeimg_classifier.skeleton import fib`,
# when using this Python module as a library.


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration for me me")
    parser.add_argument(
        "--version",
        action="version",
        version="aeimg-classifier {ver}".format(ver=__version__),
    )
    parser.add_argument(dest="n", help="n-th Fibonacci number", type=int, metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    print("The {}-th Fibonacci number **is** {}".format(args.n, fib(args.n)))
    _logger.info("Script ends here")

@click.command()
@click.option(
    "verbose",
    "-v",
    "--verbose",
    default = False,
    is_flag = True,
    type=click.BOOL
)
@click.option(
    "architecture",
    "-a",
    "--arch",
    default = 'demo',
    type=click.Choice(['demo', 'dummy', 'basic', '...'])
)
@click.option(
    "image_type",
    "-t",
    "--image-type",
    default = 'pets',
    type=click.Choice(['pets', 'numbers', 'fashion', 'homes'])
)
@click.option(
    "tags_file",
    "--tags-file",
    type=click.Path(exists=True, file_okay=True, readable=True)
)
@click.argument(
    "images_dir",
    type=click.Path(exists=True, dir_okay=True, readable=True),
    required=1
)

def run(images_dir, verbose, architecture, image_type, tags_file):

    """ I'll try to guess if you're a cute 🐱 or a fierce 🐶.

            And as an extra 🥣 I'll see if I can separate all pets into them into category and breed.

        In the near future 🚆 I will:

            * [ ] `-t numbers` Classify simple ⨐ numbers.

            * [ ] `-t fashion` Classify simple fashion 👚 clothes.

            * [ ] `-t homes`  Classify my dad's pictures of his 🏡 architect work.
    """
    print(f"🐼 I'm a panda!, this are the arguments {[images_dir, verbose, architecture, image_type, tags_file]}")

if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m aeimg_classifier.skeleton 42
    #
    run()

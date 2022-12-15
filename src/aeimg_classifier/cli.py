import argparse
import logging
import sys
import click

from aeimg_classifier import __version__

__author__ = "aemonge"
__copyright__ = "aemonge"
__license__ = "MIT"

# _logger = logging.getLogger(__name__)

# ---- Python API ----
from aeimg_classifier.lib.dummy import dummy


# ---- CLI: Helpers ----
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
        version=f"aeimg-classifier {__version__}"
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

# ---- CLI: Arguments ----
# ---- CLI: Arguments ----
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

# ---- CLI: Main ----
def run(images_dir, verbose, architecture, image_type, tags_file):

    """ I'll try to guess if you're a cute 🐱 or a fierce 🐶.

            And as an extra 🥣 I'll see if I can separate all pets into them into category and breed.

        In the near future 🚆 I will:

            * [ ] `-t numbers` Classify simple ⨐ numbers.

            * [ ] `-t fashion` Classify simple fashion 👚 clothes.

            * [ ] `-t homes`  Classify my dad's pictures of his 🏡 architect work.
    """
    dummy()
    # print(f"🐼 I'm a panda!, this are the arguments {[images_dir, verbose, architecture, image_type, tags_file]}")

if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    SEE: https://docs.python.org/3/library/__main__.html
    pass

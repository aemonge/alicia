import argparse # pylint: disable=missing-module-docstring
import logging
import sys
import click

from aeimg_classifier import __version__

__author__ = "aemonge"
__copyright__ = "aemonge"
__license__ = "MIT"

# _logger = logging.getLogger(__name__)

# ---- Python API ----
from models.dummy import DummyModel
from models.basic import BasicModel
from models.cat import Cat
from lib.dispaly_analytics import DispalyAnalytics as print_da
from lib.torchvision_downloader import TorchvisionDownloader

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
  parser.add_argument("--version", action="version", version=f"aeimg-classifier {__version__}")
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
  logging.basicConfig(level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


# ---- CLI: Arguments ----
architecture_opts = ['demo', 'dummy', 'basic', 'cat', 'download_mnist_num', '...']
@click.command()
@click.option("graph", "-g", "--graph", default=True, is_flag=True, type=click.BOOL)
@click.option("verbose", "--verbose", default=False, is_flag=True, type=click.BOOL)
@click.option("architecture", "-a", "--arch", default='dummy', type=click.Choice(architecture_opts))
@click.option("image_type",
              "-t",
              "--image-type",
              default='pets',
              type=click.Choice(['pets', 'numbers', 'fashion', 'homes']))
@click.option("tags_file", "--tags-file", type=click.Path(exists=True, file_okay=True, readable=True))
@click.argument("images_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
# ---- CLI: Main ----
def run(images_dir, graph, verbose, architecture, image_type, tags_file):
  """
  I'll try to guess if you're a cute 🐱 or a fierce 🐶, or what type of 🏘️ you should be.

    And as an extra 🥣 I'll see if I can separate all pets into them into category and breed.

  In the near future 🚆 I will:

    * [ ] `-t numbers` Classify simple mnist ⨐ numbers.

    * [ ] `-t fashion` Classify simple mnist fashion 👚 clothes.

    * [ ] `-t pets`  Classify my dad's pictures of his 🐱 architect work.

    * [ ] `-t homes`  Classify my dad's pictures of his 🏡 architect work.

  """
  if architecture == 'dummy':
    model = DummyModel(show_graph=graph, step_print_fn=print_da())
  if architecture == 'basic':
    model = BasicModel(data_dir=images_dir, step_print_fn=print_da(), verbose=verbose)
    model.splitData()
    # model.shapes()
  if architecture == 'cat':
    model = Cat(data_dir=images_dir)
  if architecture == 'download_mnist_num':
    TorchvisionDownloader(image_dir = images_dir, dataset = 'MNIST').call()

  if architecture != 'download_mnist_num':
    model.train()
    model.preview(3)


if __name__ == "__main__":
  # ^  This is a guard statement that will prevent the following code from
  #    being executed in the case someone imports this file instead of
  #    executing it as a script.
  #    SEE: https://docs.python.org/3/library/__main__.html
  pass

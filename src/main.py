"""
  A CLI to download, train, test, predict and compare an image classifiers.

  Supporting mostly all torch-vision neural networks and datasets.

  This will also identify cute 🐱 or a fierce 🐶, also flowers
  or what type of 🏘️ you should be.
"""

from dependencies.core import click
from commands import download, create, train, test, info, predict, compare

__author__ = "aemonge"
__copyright__ = "aemonge"
__license__ = "MIT"
__name__ = "alicia"
__version__ = "0.0.9"

@click.group()
@click.pass_context
@click.option("-v", "--verbose", default=False, is_flag=True, type=click.BOOL)
@click.option("-g", "--gpu", default=False, is_flag=True, type=click.BOOL)
@click.version_option(version=__version__, package_name=__name__)
def call(ctx, verbose, gpu):
  """
    A CLI to download, train, test, predict and compare an image classifiers.

    Supporting mostly all torch-vision neural networks and datasets.

    This will also identify cute 🐱 or a fierce 🐶, also flowers
    or what type of 🏘️ you should be.
  """
  ctx.ensure_object(dict)
  ctx.obj['verbose'] = verbose
  ctx.obj['use_gpu'] = gpu

call.add_command(download)
call.add_command(create)
call.add_command(train)
call.add_command(test)
call.add_command(predict)
call.add_command(info)
call.add_command(compare)

if __name__ == '__main__':
  # ^  This is a guard statement that will prevent the following code from
  #    being executed in the case someone imports this file instead of
  #    executing it as a script.
  #    SEE: https://docs.python.org/3/library/__main__.html
  pass

import click

from cli.train import train
from cli.download import download
from cli.test import test

__author__ = "aemonge"
__copyright__ = "aemonge"
__license__ = "MIT"
__name__ = "aeimg_classifier"
__version__ = "0.0.1"


@click.group()
@click.pass_context
@click.option("-v", "--verbose", default=False, is_flag=True, type=click.BOOL)
@click.version_option(version=__version__, package_name=__name__)
def call(ctx, verbose):
  """
    A CLI to download, train and test an image classifier.

    This will also identify cute 🐱 or a fierce 🐶, or what type of 🏘️ you should be.
  """
  ctx.ensure_object(dict)
  ctx.obj['verbose'] = verbose


call.add_command(download)
call.add_command(train)
call.add_command(test)

if __name__ == '__main__':
  # ^  This is a guard statement that will prevent the following code from
  #    being executed in the case someone imports this file instead of
  #    executing it as a script.
  #    SEE: https://docs.python.org/3/library/__main__.html
  pass

from cli.constants import ARCHITECTURES

from models.dummy import DummyModel
from models.basic import BasicModel
from models.cat import Cat

import click


@click.command()
@click.pass_context
@click.argument("architecture", default='dummy', type=click.Choice(ARCHITECTURES), required=1)
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.option("-n", "--number-images", default=3, type=click.INT)
def test(ctx, architecture, data_dir, number_images):
  """
    Tests the given architecture, by displaying the results for image analysis with its label probabilities.

    Use -n to increment the number of images to display
  """
  model = None
  verbose = ctx.obj['verbose']

  if architecture == 'dummy':
    model = DummyModel()
  elif architecture == 'basic':
    model = BasicModel(data_dir=data_dir, verbose=verbose)
  elif architecture == 'cat':
    model = Cat(data_dir=data_dir)
  else:
    print('Not implemented, yet 🐼')

  model.train()
  model.preview(number_images)

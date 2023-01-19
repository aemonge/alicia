from cli.constants import ARCHITECTURES
from models.dummy import DummyModel
from models.basic import BasicModel
from models.cat import Cat

import click

@click.command()
@click.pass_context
@click.argument("architecture", default='dummy', type=click.Choice(ARCHITECTURES), required=1)
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
def train(ctx, architecture, data_dir, verbose):
  """
    Train a given architecture with a data directory containing the images and a `labels.csv` file.
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

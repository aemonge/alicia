# pylint: disable=too-many-arguments
import click

from cli.constants import ARCHITECTURES
from models.dummy import DummyModel
from models.basic import BasicModel
from models.cat import Cat

@click.command()
@click.pass_context
@click.argument("architecture", default='dummy', type=click.Choice(ARCHITECTURES), required=1)
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.option("-e", "--epochs", default=3, type=click.INT)
@click.option("-s", "--save-model-file", type=click.Path(file_okay=True, exists=True, readable=True))
@click.option("-i", "--input-model-file", type=click.Path(file_okay=True, exists=True, writable=True))
def train(ctx, architecture, data_dir, save_model_file, input_model_file, epochs):
  """
    Train a given architecture with a data directory containing a '/test' and '/train' subfolder
    each with the images files and a `labels.csv` file.

    Parameters
    ----------
      ctx: click.Context
        Click context object.
      architecture: str
        The architecture to be used to train, can be a home made or based on an existing.
      data_dir: str
        The path to the data directory containing the '/test' and '/train' subfolders.
      save_model_file: str
        The path to the file where the trained model will be saved.
      input_model_file: str
        The path to the file where the trained model will be loaded.
      epochs: int
        The number of epochs to train the model for.

    Returns
    -------
      None

  """
  model = None
  verbose = ctx.obj['verbose']

  if architecture == 'dummy':
    model = DummyModel()
  elif architecture == 'basic':
    model = BasicModel(data_dir=data_dir, verbose=verbose, model_file = input_model_file, epochs=epochs)
  elif architecture == 'cat':
    model = Cat(data_dir=data_dir)
  else:
    print('Not implemented, yet 🐼')

  model.train()
  if save_model_file:
    print(save_model_file)
    model.save(save_model_file)

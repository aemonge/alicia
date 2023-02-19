from dependencies.core import click, inspect
from .shared import labels_reader
from modules import models
MODELS_NAMES = [ name for name, _ in inspect.getmembers(models, predicate=inspect.isclass) ]

@click.command()
@click.pass_context
@click.argument("architecture", default=MODELS_NAMES[0], type=click.Choice(MODELS_NAMES), required=1)
@click.argument("categories-file", type=click.Path(file_okay=True, writable=True), required=1)
@click.argument("save-file", type=click.Path(file_okay=True, writable=True), required=1)
@click.option('-d', '--dropout', type=click.FLOAT, default=0.0)
@click.option('-i', '--input-size', type=click.INT, default=28)
def create(_, architecture: str, categories_file: str, save_file: str, input_size: int, dropout: float) -> None:
  """
    Creates a new model for a given architecture.
    This will generate a .pth file to use later to train, test, and evaluate the model.
  """
  sorted_labels = labels_reader(categories_file)
  model = getattr(models, architecture)(**{
    "labels": sorted_labels, "input_size": input_size, "dropout": dropout
  })

  model.save(save_file)
  print('💚')

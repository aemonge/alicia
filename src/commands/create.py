from dependencies.core import click, inspect, Image
from .shared import labels_reader
from modules import models
MODELS_NAMES = [ name for name, _ in inspect.getmembers(models, predicate=inspect.isclass) ]

@click.command()
@click.pass_context
@click.argument("architecture", default=MODELS_NAMES[0], type=click.Choice(MODELS_NAMES), required=1)
@click.argument("categories-file", type=click.Path(file_okay=True, writable=True), required=1)
@click.argument("save-file", type=click.Path(file_okay=True, writable=True), required=1)
@click.option('-d', '--dropout', type=click.FLOAT, default=0.0)
@click.option('-a', '--from-image', type=click.Path(file_okay=True, readable=True),
  help="set the input-size from an image, resulting in {width X height}"
)
def create(_, architecture: str, categories_file: str, save_file: str,
           dropout: float, input_size: int, from_image: str) -> None:
  """
    Creates a new model for a given architecture.
    This will generate a .pth file to use later to train, test, and evaluate the model.
  """
  if from_image:
    img = Image.open(from_image)
    input_size = img.size[0] * img.size[1] * len(img.split()) # Check channels, usually L|RGB|RGBA
  sorted_labels = labels_reader(categories_file)
  kwargs = { "labels": sorted_labels, "input_size": input_size }

  if dropout > 0.0:
    kwargs["dropout"] = dropout

  model = getattr(models, architecture)(**kwargs)
  model.save(save_file)
  print('💚')

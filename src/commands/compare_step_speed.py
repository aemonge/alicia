from dependencies.core import click, torch, os
from .shared import labels_reader
from modules import models as Models
from features import Comparer, Trainer

@click.command()
@click.pass_context
@click.option("-b", "--batch-size", type=int, default=16, help="Image loader batch size")
@click.option("-l", "--learning-rate", default=round(1/137, 6), type=click.FLOAT)
@click.option("-m", "--momentum", type=click.FLOAT)
@click.argument("models_files", nargs=-1, type=click.Path(file_okay=True, exists=True, readable=True), required=1)
def step_speed(_, batch_size, learning_rate, momentum, models_files):
  """
    Compares two models performance by doing a single epoch of training.
    Using the last module for meta-data (categories, data directory, transforms)
    Like running `alicia train -p -e 1` twice with a nice diff.
  """
  models = []
  data = {}
  for model_file in models_files:
    data = torch.load(model_file)
    architecture = data["name"]
    del data["name"]
    model = getattr(Models, architecture)(**data)
    models.append(model)

  data_dir = data['data_paths']
  categories_file = data['data_paths']['labels_map']
  labels: dict = labels_reader(categories_file, _sorted=False) # pyright: ignore [reportGeneralTypeIssues]

  c = Comparer(Trainer, models, names=[ os.path.basename(n) for n in models_files])
  if momentum is not None:
    c.training(data_dir, labels, batch_size, learning_rate = learning_rate, momentum = momentum)
  else:
    c.training(data_dir, labels, batch_size, learning_rate = learning_rate)

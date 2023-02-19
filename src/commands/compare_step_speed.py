from dependencies.core import click, torch, os
from .shared import labels_reader
from modules import models as Models
from features import Comparer, Trainer

@click.command()
@click.pass_context
@click.option("-d", "--data-dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.option("-c", "--categories-file", type=click.Path(file_okay=True, writable=True), required=1)
@click.option("-b", "--batch-size", type=int, default=16, help="Image loader batch size")
@click.option("-l", "--learning-rate", default=round(1/137, 6), type=click.FLOAT)
@click.option("-m", "--momentum", type=click.FLOAT)
@click.argument("models_files", nargs=-1, type=click.Path(file_okay=True, exists=True, readable=True), required=1)
def step_speed(_, data_dir, categories_file, batch_size, learning_rate, momentum, models_files):
  """
    Compares two models performance by doing a single epoch of training.
    Like running `alicia train -p -e 1` twice with a nice diff.
  """
  labels: dict = labels_reader(categories_file, _sorted=False) # pyright: ignore [reportGeneralTypeIssues]
  models = []
  for model_file in models_files:
    data = torch.load(model_file)
    model = getattr(Models, data['name'])(**{"data": data})
    models.append(model)

  c = Comparer(Trainer, models, names=[ os.path.basename(n) for n in models_files])
  if momentum is not None:
    c.training(data_dir, labels, batch_size, learning_rate = learning_rate, momentum = momentum)
  else:
    c.training(data_dir, labels, batch_size, learning_rate = learning_rate)

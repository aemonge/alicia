from dependencies.core import click, torch, os
from .shared import labels_reader
from modules import models as Models
from features import Comparer, Trainer

@click.command()
@click.pass_context
@click.option("-b", "--batch-size", type=int, default=16, help="Image loader batch size")
@click.argument("models_files", nargs=-1, type=click.Path(file_okay=True, exists=True, readable=True), required=1)
def accuracy(_, batch_size, models_files):
  """
    Accuracy of the models, with the total time given.
    Using the last module for meta-data (categories, data directory, transforms)
    Like running `alicia test` twice with a nice diff.
  """
  models = []
  data = {}
  for model_file in models_files:
    data = torch.load(model_file)
    architecture = data['name']
    del data['name']
    model = getattr(Models, architecture)(**data)
    models.append(model)


  categories_file = data['data_paths']['labels_map']
  data_dir = data['data_paths']
  labels: dict = labels_reader(categories_file, _sorted=False) # pyright: ignore [reportGeneralTypeIssues]

  c = Comparer(Trainer, models, names=[ os.path.basename(n) for n in models_files])
  c.accuracy(data_dir, labels, batch_size)

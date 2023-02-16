from dependencies.core import click, torch, os
from .shared import labels_reader
from modules import models as Models
from features import Comparer

@click.command()
@click.pass_context

@click.argument("models_files", nargs=-1)
def diff_info(_, models_files):
  """
    Compares the models described in the given files.
  """
  models = []
  for model_file in models_files:
    data = torch.load(model_file)
    model = getattr(Models, data['name'])(data)
    model.load(model_file)
    models.append(model)

  c = Comparer(models, names=[ os.path.basename(n) for n in models_files])
  print(c)

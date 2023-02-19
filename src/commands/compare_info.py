from dependencies.core import click, torch, os
from modules import models as Models
from features import Comparer, Trainer

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
    model = getattr(Models, data['name'])(**{"data": data})
    models.append(model)

  c = Comparer(Trainer, models, names=[ os.path.basename(n) for n in models_files])
  print(c)

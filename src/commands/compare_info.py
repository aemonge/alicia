from dependencies.core import click, torch, os
from dependencies.datatypes import AbsModule
from modules.models import Basic
from features import Comparer

@click.command()
@click.pass_context

@click.argument("models_files", nargs=-1)
def diff_info(_, models_files):
  """
    Compares the models described in the given files.
  """
  models: list[AbsModule] = []
  data = []

  for model_file in models_files:
    data = torch.load(model_file)

    match data['name'].lower():
      case 'basic':
        model = Basic(data)
      case _:
        raise ValueError(f'Unknown model: {data["name"]}')

    model.load(model_file)
    models.append(model)

  c = Comparer(models, names=[ os.path.basename(n) for n in models_files])
  print(c)

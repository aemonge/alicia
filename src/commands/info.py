from dependencies.core import click, torch
from modules import models

@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
def info(_, model_file):
  """
    Display information about a model architecture.
    Hidden layers, the trained time, out put, and features.
  """
  data = torch.load(model_file)
  model = getattr(models, data['name'])(**{"data": data})

  print(model)

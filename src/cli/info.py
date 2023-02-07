import click
import torch

from models.Basic import Basic
from models.abstract import Module

@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
def info(ctx, model_file):
  """
    Display information about a model architecture.
    Hidden layers, the trained time, out put, and features.
  """
  data = torch.load(model_file)

  model: Module
  match data['name'].lower():
    case 'basic':
      model = Basic(data)
    case _:
      raise ValueError(f'Unknown model: {data["name"]}')

  model.load(model_file)
  print(model)

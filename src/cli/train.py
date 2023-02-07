import click
import torch

from models.dummy import DummyModel
from models.basic import BasicModel

from models.Basic import Basic
from models.abstract import Module
from transforms.basic import transforms
from actions.train import Trainer

from models.cat import Cat

@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.option("-b", "--batch-size", type=int, default=16, help="Image loader batch size")
@click.option("-e", "--epochs", default=3, type=click.INT)
@click.option("-p", "--pretend", default=False, type=click.BOOL, is_flag=True)
def train(ctx, model_file, data_dir, batch_size, epochs, pretend):
  """
    Train a given architecture with a data directory containing a '/validate' and '/train' subfolder
    each with the images files and a `labels.csv` file.
  """
  model: Module
  data = torch.load(model_file)
  verbose = ctx.obj['verbose']

  match data['name'].lower():
    case 'basic':
      model = Basic(data)
    case _:
      raise ValueError(f'Unknown model: {data["name"]}')

  model.load(model_file)
  trainer = Trainer(model, transforms)
  trainer.train(data_dir, batch_size, epochs)

  if not pretend:
    model.save(model_file)

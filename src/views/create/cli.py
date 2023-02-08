import click

import csv

from constants import ARCHITECTURES
from modules.models.Basic import Basic
from modules.models.AbsModule import AbsModule


@click.command()
@click.pass_context
@click.argument("architecture", default='dummy', type=click.Choice(ARCHITECTURES), required=1)
@click.argument("categories-file", type=click.Path(file_okay=True, writable=True), required=1)
@click.argument("save-file", type=click.Path(file_okay=True, writable=True), required=1)
@click.option('-u', '--hidden-units', type=click.INT, default=8)
@click.option('-d', '--dropout', type=click.FLOAT, default=0.01)
@click.option('-i', '--initial-input-size', type=click.INT, default=28)

def create(ctx, architecture: str, categories_file: str, save_file: str,
           hidden_units: int, initial_input_size: int, dropout: float) -> None:
  """
    Creates a new model for a given architecture.
    This will generate a .pth file to use later to train, test, and evaluate the model.
  """
  model: AbsModule
  verbose = ctx.obj['verbose']

  labels = set()
  with open(categories_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for _, label in reader:
      labels.add(label)
  sorted_labels = sorted(list(labels))

  match architecture.lower():
    case 'basic':
      model = Basic(sorted_labels)

  model.create(input_size=initial_input_size, hidden_units=hidden_units, dropout=dropout)
  model.save(save_file)

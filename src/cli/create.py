import click

import csv

from cli.constants import ARCHITECTURES
from models.Basic import Basic
from models.abstract import Module


@click.command()
@click.pass_context
@click.argument("architecture", default='dummy', type=click.Choice(ARCHITECTURES), required=1)
@click.argument('hidden_units', type=click.INT, default=8, required=1)
@click.argument("arch-path", type=click.Path(file_okay=True, writable=True), required=1)
@click.option('-c', '--category-labels', type=click.Path(exists=True, file_okay=True, readable=True), required=1,
              help="category labels csv file, which contains the `file_name, category names` per row"
              )
@click.option("-l", "--learning-rate", default=(1/137), type=click.FLOAT)
@click.option("-m", "--momentum", default=0.9, type=click.FLOAT)

def create(ctx, architecture: str, hidden_units, arch_path, category_labels, learning_rate, momentum):
  """
    Creates a new model for a given architecture.
    This will generate a .pth file to use later to train, test, and evaluate the model.
  """
  model: Module
  verbose = ctx.obj['verbose']

  labels = set()
  with open(category_labels, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for _, label in reader:
      labels.add(label)

  match architecture.lower():
    case 'basic':
      model = Basic(list(labels), learning_rate, momentum)

  model.create(784, hidden_units)
  model.save(arch_path)

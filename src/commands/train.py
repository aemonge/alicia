from dependencies.core import csv, click, torch
from dependencies.fancy import colored
from dependencies.datatypes import AbsModule
from modules.models import *
from modules.transforms import ImageTransforms
from features import Trainer

@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.argument("categories-file", type=click.Path(file_okay=True, writable=True), required=1)
@click.option("-b", "--batch-size", type=int, default=16, help="Image loader batch size")
@click.option("-e", "--epochs", default=3, type=click.INT)
@click.option("-l", "--learning-rate", default=round(1/137, 6), type=click.FLOAT)
@click.option("-m", "--momentum", type=click.FLOAT)
@click.option("-p", "--pretend", default=False, type=click.BOOL, is_flag=True)
def train(ctx, model_file, data_dir, categories_file, batch_size, epochs, learning_rate, momentum, pretend):
  """
    Train a given architecture with a data directory containing a '/validate' and '/train' subfolder
    each with the images files and a `labels.csv` file.
  """
  data = torch.load(model_file)

  labels = {}
  with open(categories_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for filename, label in reader:
      labels[filename] = label

  model: AbsModule
  match data['name'].lower():
    case 'basic':
      model = Basic(data)
    case _:
      raise ValueError(f'Unknown model: {data["name"]}')

  model.load(model_file)
  if pretend:
    print(colored(' Results of the training will not saved, since we are just pretending\n', 'yellow'))

  if momentum is not None:
    trainer = Trainer(model, ImageTransforms, learning_rate = learning_rate, momentum = momentum)
  else:
    trainer = Trainer(model, ImageTransforms, learning_rate = learning_rate)

  trainer.train(data_dir, labels, batch_size, epochs)

  if not pretend:
    model.save(model_file)

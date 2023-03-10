from dependencies.core import click, torch
from dependencies.fancy import colored
from .shared import labels_reader
from modules import models, transforms
from features import Trainer

@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
@click.option("-b", "--batch-size", type=int, default=16, help="Image loader batch size")
@click.option("-e", "--epochs", default=3, type=click.INT)
@click.option("-l", "--learning-rate", default=round(1/137, 6), type=click.FLOAT)
@click.option("-m", "--momentum", type=click.FLOAT)
@click.option("-p", "--pretend", default=False, type=click.BOOL, is_flag=True)
@click.option("-f", "--freeze-parameters", default=False, type=click.BOOL, is_flag=True)
def train(_, model_file, batch_size, epochs, learning_rate, momentum, pretend, freeze_parameters):
  """
    Train a given architecture with a data directory containing a '/validate' and '/train' subfolder
    each with the images files and a `labels.csv` file.
  """
  data = torch.load(model_file)
  architecture = data["name"]
  data_dir = data['data_paths']
  categories_file = data['data_paths']['labels_map']
  del data["name"]

  model = getattr(models, architecture)(**data)
  transform = getattr(transforms, data['transform'])()
  labels: dict = labels_reader(categories_file, _sorted=False)

  if pretend:
    print('\n', colored(' Results of the training will not saved, since we are just pretending', 'yellow'))

  if momentum is not None:
    trainer = Trainer(model, transform, learning_rate = learning_rate, momentum = momentum)
  else:
    trainer = Trainer(model, transform, learning_rate = learning_rate)

  trainer.train(data_dir, labels, batch_size, epochs, freeze_parameters = freeze_parameters)

  if not pretend:
    model.save(model_file)

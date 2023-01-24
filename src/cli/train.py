# pylint: disable=too-many-arguments
import click

from cli.constants import ARCHITECTURES
from models.dummy import DummyModel
from models.basic import BasicModel
from models.cat import Cat

@click.command()
@click.pass_context
@click.argument("architecture", default='dummy', type=click.Choice(ARCHITECTURES), required=1)
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.option("-e", "--epochs", default=3, type=click.INT)
@click.option("-g", "--gpu-enabled", default=False, type=click.BOOL, is_flag=True)
@click.option("-s", "--save-model-file", type=click.Path(file_okay=True, writable=True))
@click.option("-i", "--input-model-file", type=click.Path(file_okay=True, exists=True, readable=True))
def train(ctx, architecture, data_dir, save_model_file, input_model_file, epochs, gpu_enabled):
  """
    Train a given architecture with a data directory containing a '/test' and '/train' subfolder
    each with the images files and a `labels.csv` file.
  """
  model = None
  verbose = ctx.obj['verbose']

  if architecture == 'dummy':
    model = DummyModel()
  elif architecture == 'basic':
    model = BasicModel(data_dir=data_dir, verbose=verbose, model_file=input_model_file, epochs=epochs, use_gpu=gpu_enabled)
  elif architecture == 'cat':
    model = Cat(data_dir=data_dir)
  else:
    print('Not implemented, yet 🐼')

  model.train()
  if save_model_file:
    print(save_model_file)
    model.save(save_model_file)

# pylint: disable=too-many-arguments
import click

from cli.constants import ARCHITECTURES
from models.dummy import DummyModel
from models.basic import BasicModel
from models.cat import Cat

@click.command()
@click.pass_context
@click.argument("architecture", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.option('-c', '--console-plot', default=False, type=click.BOOL, is_flag=False)
@click.option('--tilted-title', default=False, type=click.BOOL, is_flag=False)
@click.option("-n", "--n-images-test", default=1, type=click.INT,
  help="Increment the number of images to display with the class bar chart and images preview"
)
def test(ctx, architecture, data_dir, console_plot, tilted_title, n_images_test):
  """
    Test a pre trained model. It will look for the `./test` folder inside the data directory.
    A file named `./labels.csv` should exist in the root directory of the data folder.
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

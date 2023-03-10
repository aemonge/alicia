from dependencies.core import click, torch, inspect
from .shared import labels_reader
from modules import models
from features import Trainer
from modules import transforms

TRANFORMS_NAMES = [ name for name, _ in inspect.getmembers(transforms, predicate=inspect.isfunction) ]

@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
@click.option("-b", "--batch-size", type=int, default=16, help="Image loader batch size")
@click.option('-c', '--console-plot', default=False, type=click.BOOL, is_flag=True)
@click.option('-h', '--h-title', default=False, type=click.BOOL, is_flag=True, help="Show the title horizontally")
@click.option("-n", "--n-images-test", default=0, type=click.INT,
  help="Increment the number of images to display with the class bar chart and images preview"
)
def test(_, model_file, batch_size, console_plot, h_title, n_images_test):
  """
    Test a pre trained model. It will look for the `./test` folder inside the data directory.
    A file named `./labels.csv` should exist in the root directory of the data folder.
  """
  data = torch.load(model_file)
  architecture = data["name"]
  data_dir = data['data_paths']
  categories_file = data['data_paths']['labels_map']
  del data["name"]

  model = getattr(models, architecture)(**data)
  transform = getattr(transforms, data['transform'])()
  labels: dict = labels_reader(categories_file, _sorted=False) # pyright: ignore [reportGeneralTypeIssues]

  trainer = Trainer(model, transform)
  trainer.test(data_dir, labels, batch_size)
  if n_images_test > 0:
    if console_plot:
      trainer.show_test_results_in_console(data_dir['test'], labels, n_images_test)
    else:
      trainer.show_test_results(data_dir['test'], labels, n_images_test, h_title)

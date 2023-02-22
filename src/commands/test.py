from dependencies.core import click, torch, inspect
from .shared import labels_reader
from modules import models
from features import Trainer
from modules import transforms

TRANFORMS_NAMES = [ name for name, _ in inspect.getmembers(transforms, predicate=inspect.isfunction) ]

@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.argument("categories-file", type=click.Path(file_okay=True, writable=True), required=1)
@click.option("-b", "--batch-size", type=int, default=16, help="Image loader batch size")
@click.option('-c', '--console-plot', default=False, type=click.BOOL, is_flag=True)
@click.option('-h', '--h-title', default=False, type=click.BOOL, is_flag=True, help="Show the title horizontally")
@click.option("-n", "--n-images-test", default=0, type=click.INT,
  help="Increment the number of images to display with the class bar chart and images preview"
)
@click.option("-t", "--transform-name", default=TRANFORMS_NAMES[0], type=click.Choice(TRANFORMS_NAMES))
def test(_, model_file, data_dir, categories_file, batch_size, console_plot,
         h_title, n_images_test, transform_name):
  """
    Test a pre trained model. It will look for the `./test` folder inside the data directory.
    A file named `./labels.csv` should exist in the root directory of the data folder.
  """
  labels: dict = labels_reader(categories_file, _sorted=False) # pyright: ignore [reportGeneralTypeIssues]

  data = torch.load(model_file)
  model = getattr(models, data['name'])(**{"data": data})
  transform = getattr(transforms, transform_name)()

  trainer = Trainer(model, transform)
  trainer.test(data_dir, labels, batch_size)
  if n_images_test > 0:
    if console_plot:
      trainer.show_test_results_in_console(data_dir, labels, n_images_test)
    else:
      trainer.show_test_results(data_dir, labels, n_images_test, h_title)

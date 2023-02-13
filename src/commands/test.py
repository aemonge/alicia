from dependencies.core import csv, click, torch
from modules.transforms import ImageTransforms
from modules.models import Basic
from features import Trainer

@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.argument("categories-file", type=click.Path(file_okay=True, writable=True), required=1)
@click.option("-b", "--batch-size", type=int, default=16, help="Image loader batch size")
@click.option('-c', '--console-plot', default=False, type=click.BOOL, is_flag=False)
@click.option('-h', '--h-title', default=False, type=click.BOOL, is_flag=True, help="Show the title horizontally")
@click.option("-n", "--n-images-test", default=0, type=click.INT,
  help="Increment the number of images to display with the class bar chart and images preview"
)
def test(ctx, model_file, data_dir, categories_file, batch_size, console_plot, h_title, n_images_test):
  """
    Test a pre trained model. It will look for the `./test` folder inside the data directory.
    A file named `./labels.csv` should exist in the root directory of the data folder.
  """
  labels = {}
  with open(categories_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for filename, label in reader:
      labels[filename] = label

  model: AbsModule
  data = torch.load(model_file)

  match data['name'].lower():
    case 'basic':
      model = Basic(data)
    case _:
      raise ValueError(f'Unknown model: {data["name"]}')

  model.load(model_file)
  trainer = Trainer(model, ImageTransforms)
  trainer.test(data_dir, labels, batch_size)
  if n_images_test > 0:
    trainer.show_test_results(data_dir, labels, n_images_test, h_title)

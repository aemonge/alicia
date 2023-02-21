from dependencies.core import click, torch, os, inspect
from .shared import labels_reader
from modules import models as Models
from features import Comparer, Trainer
from modules import transforms

TRANFORMS_NAMES = [ name for name, _ in inspect.getmembers(transforms, predicate=inspect.isfunction) ]

@click.command()
@click.pass_context
@click.option("-d", "--data-dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.option("-c", "--categories-file", type=click.Path(file_okay=True, writable=True), required=1)
@click.option("-b", "--batch-size", type=int, default=16, help="Image loader batch size")
@click.option("-t", "--transform-name", default=TRANFORMS_NAMES[0], type=click.Choice(TRANFORMS_NAMES))
@click.argument("models_files", nargs=-1, type=click.Path(file_okay=True, exists=True, readable=True), required=1)
def accuracy(_, data_dir, categories_file, batch_size, transform_name, models_files):
  """
    Accuracy of the models, with the total time given.
    Like running `alicia test` twice with a nice diff.
  """
  labels: dict = labels_reader(categories_file, _sorted=False) # pyright: ignore [reportGeneralTypeIssues]
  transform = getattr(transforms, transform_name)
  models = []
  for model_file in models_files:
    data = torch.load(model_file)
    model = getattr(Models, data['name'])(**{"data": data})
    models.append(model)

  c = Comparer(Trainer, models, names=[ os.path.basename(n) for n in models_files])
  c.accuracy(transform, data_dir, labels, batch_size)

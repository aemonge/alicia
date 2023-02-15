from dependencies.core import csv, click, torch, os
from dependencies.datatypes import AbsModule
from modules.models import Basic
from features import Comparer

@click.command()
@click.pass_context
@click.option("-d", "--data-dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.option("-c", "--categories-file", type=click.Path(file_okay=True, writable=True), required=1)
@click.option("-b", "--batch-size", type=int, default=16, help="Image loader batch size")
@click.argument("models_files", nargs=-1, type=click.Path(file_okay=True, exists=True, readable=True), required=1)
def accuracy(_, data_dir, categories_file, batch_size, models_files):
  """
    Accuracy of the models, with the total time given.
    Like running `alicia test` twice with a nice diff.
  """
  labels = {}
  with open(categories_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for filename, label in reader:
      labels[filename] = label

  models: list[AbsModule] = []
  data = []

  for model_file in models_files:
    data = torch.load(model_file)

    match data['name'].lower():
      case 'basic':
        model = Basic(data)
      case _:
        raise ValueError(f'Unknown model: {data["name"]}')

    model.load(model_file)
    models.append(model)

  c = Comparer(models, names=[ os.path.basename(n) for n in models_files])
  c.accuracy(data_dir, labels, batch_size)

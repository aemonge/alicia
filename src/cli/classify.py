import click

from cli.constants import ARCHITECTURES
from models.dummy import DummyModel
from models.basic import BasicModel
from models.cat import Cat


@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.option("-n", "--number-images", default=0, type=click.INT,
  help="Increment the number of images to display with the class bar chart and images preview"
)
def classify(ctx, model_file, data_dir, number_images):
  """
    Classify images using a trained model.
  """
  model = None
  verbose = ctx.obj['verbose']

  # if architecture == 'dummy':
  #   model = DummyModel()
  # elif architecture == 'basic':
    # model = BasicModel(data_dir=data_dir, verbose=verbose, model_file = model_file)
  # elif architecture == 'cat':
  #   model = Cat(data_dir=data_dir)
  # else:
  #   print('Not implemented, yet 🐼')
  model = BasicModel(data_dir=data_dir, verbose=verbose, model_file = model_file)

  csv = model.call(data_dir)
  with open(f"{data_dir}/labels.csv", 'w', encoding='utf-8') as file:
    for value in csv:
      file.write(value)
      file.write('\n')
  file.close()

  if number_images > 0:
    model.preview(image_count = number_images, path = data_dir)

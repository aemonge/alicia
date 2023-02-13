import click

from modules.models import Basic

@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.option("-k", "--top_k", default=1, type=click.INT,
  help="Show the top-k most likely categories."
)
def predict(ctx, model_file, data_dir, number_images):
  """
    Predict images using a pre trained model.
  """
  data = torch.load(model_file)

  model: AbsModule
  match data['name'].lower():
    case 'basic':
      model = Basic(data)
    case _:
      raise ValueError(f'Unknown model: {data["name"]}')

  # if architecture == 'dummy':
  #   model = DummyModel()
  # elif architecture == 'basic':
    # model = BasicModel(data_dir=data_dir, verbose=verbose, model_file = model_file)
  # elif architecture == 'cat':
  #   model = Cat(data_dir=data_dir)
  # else:
  #   print('Not implemented, yet 🐼')
  # model : AbsModule = BasicModel(data_dir=data_dir, verbose=verbose, model_file = model_file)
  #
  # csv = model.call(data_dir)
  # with open(f"{data_dir}/labels.csv", 'w', encoding='utf-8') as file:
  #   for value in csv:
  #     file.write(value)
  #     file.write('\n')
  # file.close()
  #
  # if number_images > 0:
  #   model.preview(image_count = number_images, path = data_dir)

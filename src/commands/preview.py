from dependencies.core import click, torch, inspect
from modules import transforms
from features import ImageViewer

TRANFORMS_NAMES = [ name for name, _ in inspect.getmembers(transforms, predicate=inspect.isfunction) ]

@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
@click.argument("action-type", type=click.Choice(['train', 'valid', 'test', 'display']), default='train')
@click.option("-n", "--num-images", default=3, type=click.INT, help="The number of images to preview")
def preview(_, model_file, action_type, num_images):
  """
    Preview the images transformed and the original. To understand how the network will process the data.
    It will use a random image from the defined data folder, for a given action type.
  """
  data = torch.load(model_file)
  data_dir = data['data_paths']
  if action_type == 'display':
    data_dir = data_dir['test']
  else:
    data_dir = data_dir[action_type]

  transform = getattr(transforms, data['transform'])()
  transform = transform[action_type]

  viewer = ImageViewer(data_dir, transform)
  viewer(num_images)

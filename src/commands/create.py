from dependencies.core import click, torch, urllib, tempfile, inspect, os
from .shared import labels_reader
from modules import models, transforms
MODELS_NAMES = [ name for name, _ in inspect.getmembers(models, predicate=inspect.isclass) ]
TRANFORMS_NAMES = [ name for name, _ in inspect.getmembers(transforms, predicate=inspect.isfunction) ]

help_layer_change='\
  (id, nn.module) change a classifier layer. \
    Id can be -1: Use "-1" "None" to remove the last\
    nn.module as text: "0" "Linear(784, 101")\
'

@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, readable=True, writable=True), required=1)
@click.option('-a', '--architecture', type=click.Choice(MODELS_NAMES),
              help="If the architecture is given for an existing model, it will overwrite the features."
              )
@click.option('-c', '--categories-file', type=click.Path(file_okay=True, readable=True))
@click.option("-t", "--transform-name", default=TRANFORMS_NAMES[0], type=click.Choice(TRANFORMS_NAMES))
@click.option("-D", "--data_dir", type=click.Path(exists=True, dir_okay=True, readable=True))
@click.option('-n', '--num-classes', type=click.INT, default=None)
@click.option('-i', '--input-size', type=click.INT, default=None)
@click.option('-s', '--state-dict-file', type=click.Path(file_okay=True, writable=True))
@click.option('-w', '--state-dict-weights-url', type=click.STRING,
              help="In example: https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth")
@click.option('-d', '--dropout', type=click.FLOAT, default=0.0)
@click.option('-f', '--feature-layers', type=(int, str), help=help_layer_change)
@click.option('-k', '--classifier-layers', type=(int, str), help=help_layer_change)
def create(_, model_file, architecture, categories_file, transform_name, data_dir, num_classes, input_size,
           state_dict_file, state_dict_weights_url, dropout, feature_layers, classifier_layers):
  """
    Changes the hyper parameters of a model.
    It also allows you to use pre-trained weights, by re-creating the model
  """
  kwargs = {
    'path': model_file
  }
  architecture_name = architecture
  if categories_file is not None:
    kwargs["labels"] = labels_reader(categories_file)
  if num_classes is not None:
    kwargs["num_classes"] = num_classes
  if input_size is not None:
    kwargs["input_size"] = input_size
  if dropout is not None:
    kwargs["dropout"] = dropout
  if transform_name is not None:
    kwargs["transform"] = transform_name
  if data_dir is not None:
    kwargs["data_paths"] = {
      'train': f"{data_dir}/train",
      'test': f"{data_dir}/test",
      'valid': f"{data_dir}/valid",
      'labels_map': f"{data_dir}/labels.csv",
    }

  if state_dict_file is not None:
    kwargs["state_dict"] = torch.load(state_dict_file)
  if state_dict_weights_url is not None:
    print('💙', end='\r')
    with tempfile.NamedTemporaryFile(mode="w+") as file:
      urllib.request.urlretrieve(state_dict_weights_url, file.name) # pyright: ignore [reportGeneralTypeIssues]
      kwargs["state_dict"] = torch.load(file.name)

  print('💛', end='\r')

  if os.path.exists(kwargs['path']):
    data = torch.load(kwargs['path'])
    architecture_name = architecture_name or data['name']
    for key in ['input_size', 'labels', 'num_classes', 'dropout', 'state_dict', 'features', 'classifier',
                'avgpool', 'training_history', 'transform', 'data_paths', 'path']:
      if key not in kwargs and key in data:
        kwargs[key] = data[key]

  model = getattr(models, architecture_name)(architecture is not None, **kwargs)

  if feature_layers is not None or classifier_layers is not None:
    if feature_layers is not None:
      model.modify_parameters("features", feature_layers[0], feature_layers[1])
    if classifier_layers is not None:
      model.modify_parameters("classifier", classifier_layers[0], classifier_layers[1])

  model.save()
  print('💚')

from dependencies.core import click, torch, urllib, tempfile
from dependencies.datatypes import nnModule
from .shared import labels_reader
from modules import models

help_layer_change='\
  (id, nn.module) change a classifier layer. \
    Id can be $: Use "$" "None" to remove the last\
    nn.module as text: "0" "Linear(784, 101")\
'

@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
@click.option('-r', '--re-create', is_flag=True, help="Re-create the model with passed parameters")
@click.option('-c', '--categories-file', type=click.Path(file_okay=True, writable=True))
@click.option('-n', '--num-classes', type=click.INT, default=None)
@click.option('-i', '--input-size', type=click.INT, default=None)
@click.option('-s', '--state-dict-file', type=click.Path(file_okay=True, writable=True))
@click.option('-w', '--state-dict-weights-url', type=click.STRING,
              help="In example: https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth")
@click.option('-d', '--dropout', type=click.FLOAT, default=0.0)
@click.option('-f', '--feature-layers', type=(int, str), help=help_layer_change)
@click.option('-k', '--classifier-layers', type=(int, str), help=help_layer_change)
def modify(_, model_file, re_create, categories_file, num_classes, input_size, state_dict_file,
           state_dict_weights_url, dropout, feature_layers, classifier_layers):
  """
    Changes the hyper parameters of a model.
    It also allows you to use pre-trained weights, by re-creating the model
  """
  kwargs = {}
  if categories_file is not None:
    kwargs["labels"] = labels_reader(categories_file)
  if num_classes is not None:
    kwargs["num_classes"] = num_classes
  if input_size is not None:
    kwargs["input_size"] = input_size
  if dropout is not None:
    kwargs["dropout"] = dropout

  print(feature_layers, classifier_layers)

  if state_dict_file is not None:
    kwargs["state_dict"] = torch.load(state_dict_file)
  if state_dict_weights_url is not None:
    print('💙', end='\r')
    with tempfile.NamedTemporaryFile(mode="w+") as file:
      urllib.request.urlretrieve(state_dict_weights_url, file.name) # pyright: reportGeneralTypeIssues=false
      kwargs["state_dict"] = torch.load(file.name)

  print('💛', end='\r')
  data = torch.load(model_file)
  if re_create:
    init_kwargs = {}
    for key in ['input_size', 'labels', 'num_classes', 'dropout']:
      if not key in kwargs:
        init_kwargs[key] = data[key]
      else:
        init_kwargs[key] = kwargs[key]

    model = getattr(models, data['name'])(**init_kwargs)
  else:
    model = getattr(models, data['name'])(**{"data": data})

  if feature_layers is not None:
    model.modify_parameters("features", feature_layers[0], feature_layers[1])
  if classifier_layers is not None:
    model.modify_parameters("classifier", classifier_layers[0], classifier_layers[1])
  model.modify(**kwargs)

  model.save(model_file)
  print('💚')
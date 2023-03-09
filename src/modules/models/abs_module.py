from dependencies.core import torch, ABCMeta, textwrap, time, asizeof, os, torch
from dependencies.datatypes import Parameter, Iterator
from libs import sizeof_formated, get_args_kwargs_from_string
from modules import transforms

class AbsModule(torch.nn.Module, metaclass=ABCMeta):
  def __init__(self, *, training_history: list|None = None, state_dict: dict|None = {},
               labels:list = [], input_size: int = 64, dropout: float = 0.0, num_classes: int|None = None,
               momentum: float = 0.1, transform = None, data_paths: dict|None = None, path: str|None = None,
               features: torch.nn.Module|None = None, classifier: torch.nn.Module|None = None,
               avgpool: torch.nn.Module|None = None
               ) -> None:
    """
      Constructor of the neural network.

      Parameters:
      -----------
        state_dict: dict|None
          The state dict to load the model from.
        training_history: list
          A list of training history (pre-trained).
        labels: list
          A list of labels.
        input_size: int
          The input size.
        dropout: float
          The dropout probability.
        num_classes: int|None
          The number of classes, to use a output.
        momentum: float
          The momentum of the Batch Normalization (when applicable).
        transform: torch.nn.Module
          The transformation to apply to the training or testing data.
        data_paths: dict|None
          The paths to the data, usually the train, test, valid folders.
        path: str|None
          The path to save the model.
        features: torch.nn.Module|None
          The features of the neural network (pre-created).
        classifier: torch.nn.Module|None
          The classifier of the neural network (pre-created).
        avgpool: torch.nn.Module|None
          The avgpool of the neural network (pre-created).
    """
    super().__init__()
    self.labels = labels
    self.num_classes = len(labels) if num_classes is None else num_classes
    self.training_history = []
    self.input_size = input_size
    self.dropout = dropout
    self.momentum = momentum
    self.transform = transform
    self.data_paths = data_paths
    self.path = path

    if training_history is not None:
      self.training_history = training_history
    if features is not None:
      self.features = features
    if classifier is not None:
      self.classifier = classifier
    if avgpool is not None:
      self.avgpool = avgpool
    if len(state_dict) > 0:
      self.load_state_dict(state_dict)

  def __str__(self):
    """
      A verbose string representation of the neural network.

      Returns:
      --------
        : str
          meta, data_paths, labels, features, classifier, training_history
    """
    features_str = "\n  ".join(str(self.features).split("\n"))
    labels_str = textwrap.fill(str(self.labels)[1:-1], width=80)
    labels_str = labels_str.replace('\n', '\n' + ' '*4)

    formated_size = sizeof_formated(asizeof.asizeof(self))
    formated_disk_size = 'E: [UnReachable]'
    if os.path.isfile(self.path):
      formated_disk_size = sizeof_formated(os.path.getsize(self.path))

    classifier_str = ''
    size_str = f"size (memory): {formated_size},\tsize (disk): {formated_disk_size}" + \
        f",\tstate dict len: {len(self.state_dict())}"
    train_str = f"train folder: {self.data_paths['train']},,\tvalid folder: {self.data_paths['valid']}"+ \
      f"\n{' '*4}labels map file: {self.data_paths['labels_map']},\ttest folder: {self.data_paths['test']}"

    transform_spec = getattr(transforms, self.transform)()
    transform_str = f"{self.transform}\n"

    for val in ['train', 'valid', 'test', 'display']:
      transform_str += f"{' '*4}({val}):\n"
      ix = 0
      for trans in transform_spec[val].transforms:
        transform_str += f"{' '*6}({ix}) {str(trans)}\n"
        ix += 1
    transform_str = transform_str[:-1]

    if hasattr(self, "classifier"):
      classifier_str = "\n  ".join(str(self.classifier).split("\n"))

    training_history_str = ""*4
    for line in self.training_history: # pyright: reportGeneralTypeIssues=false
      time_f = time.strftime("%H:%M:%S", time.gmtime(line[3][1]))
      training_history_str += f"Accuracy: {line[4][1]:.2f}%, "
      training_history_str += f"Time: {time_f}, "

      training_history_str += f"Epochs: {line[0][1]}, "
      training_history_str += f"Batch: {line[1][1]}, "
      training_history_str += f"Items: ({line[2][1]}, {line[2][2]})\n"

      if (len(line) > 5): # Retro-Compatibillity TODO: Will drop on refactor
        training_history_str += ' '*6 + f"Criterion: {line[5][1]},\n"
        # optmi_str = self._optimizer_str(self.optimizer)
        optmi_str = str(line[6][1]).replace(' ', '').replace('\n', ', ').replace(':', '=')
        optmi_str = optmi_str.replace(', )', '').replace('(, ', '(')
        for i in [70, 140, 190]:
          optmi_str = optmi_str[:i] + '\n' + ' ' * 21 + optmi_str[i:]
        training_history_str += ' '*6 +  f"Optimizer: {optmi_str}"
      training_history_str += '\n' + ' ' * 4

    training_history_str = training_history_str[:-5] # 4 will be a constant

    return f"{self.__repr__()} @ ./{self.path} " + '{ \n' + \
      f"  (size): \n    {size_str}\n" + \
      f"  (transforms): {transform_str}\n" + \
      f"  (data paths): \n    {train_str}\n" + \
      f"  (labels): \n    {labels_str}\n" + \
      f"  (features): {features_str}\n" + \
      (f"  (classifier): {classifier_str}\n" if hasattr(self, 'classifier') else '') + \
      (f"  (training history): \n    {training_history_str[:-1]}" if len(self.training_history) > 0 else '')

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    """
      A forward pass of the neural network.

      Parameters:
      -----------
        x: torch.Tensor
          A batch of input features.

      Returns:
      --------
        torch.Tensor
    """
    return self.forward(x)

  def parameters(self) -> Iterator[Parameter]:
    """
      Get the parameters of the neural network.

      Returns:
      --------
        Iterator[Parameter]
    """
    return self.features.parameters()

  def modify_parameters(self, parameter_type: str, idx: int, module: str) -> None:
    """
      Modify the classifier or features of the neural network.

      Parameters:
      -----------
        parameter_type: str
          The type of the parameter to modify either classifier or features.
        idx: str|int
          The index of the classifier to modify. Or -1 for the last
        module: str (torch.nn.Module)
          The new classifier or feature.

      Returns:
      --------
        None
    """
    if parameter_type != 'classifier' and parameter_type != 'features':
      raise ValueError(f"parameter_type must be either 'classifier' or 'features', not {parameter_type}")

    parameters = list(getattr(self, parameter_type))

    if module != "None":
      func_name, fargs, fkwargs = get_args_kwargs_from_string(module)
      mod = getattr(torch.nn, func_name)(*fargs, **fkwargs)
      parameters.insert(idx, mod)
    else:
      del parameters[idx]

    setattr(self, parameter_type, torch.nn.Sequential(*parameters))

  def save(self, path: str|None = None) -> None:
    """
      Save the neural network.

      Parameters:
      -----------
        path: str
          path to save model

      Returns:
      --------
        None
    """
    obj = {
      'name': self.__repr__()[:-2], # Trim the () ex. Basic()/Basic
      'input_size': self.input_size,
      'num_classes': self.num_classes,
      'labels': self.labels,
      'features': self.features,
      'state_dict': self.state_dict(),
    }

    if hasattr(self, 'training_history'):
      obj['training_history'] = self.training_history
    if hasattr(self, 'dropout') and self.dropout > 0.0:
      obj['dropout'] = self.dropout
    if hasattr(self, 'momentum') and self.momentum != 0.1:
      obj['momentum'] = self.momentum
    if hasattr(self, 'classifier'):
      obj['classifier'] = self.classifier
    if hasattr(self, 'avgpool'): # Supporting AlexNet
      obj['avgpool'] = self.avgpool
    if hasattr(self, 'transform'):
      obj['transform'] = self.transform
    if hasattr(self, 'data_paths'):
      obj['data_paths'] = self.data_paths
    if hasattr(self, 'path'):
      obj['path'] = self.path
    else:
      obj['path'] = path

    torch.save(obj, obj['path'])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
      A forward pass of the neural network.

      Parameters:
      -----------
        x: torch.Tensor
          A batch of input features.

      Returns:
      --------
        torch.Tensor

      Help:
      -----
        model.forward = lambda x: model.classifier(model.features(x)).view(x.size(0), class_count)
    """
    x = self.features(x)
    x = self.classifier(x)
    return x

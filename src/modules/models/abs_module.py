from dependencies.core import torch, abstractmethod, abstract_attribute, ABCMeta, textwrap, sys, time

# TODO: Move this as a library
def sizeof_formated(num, suffix='B'):
    """
      Converts a size in bytes to a human-readable format.

      Parameters:
      -----------
        num: int
          Size in bytes.
        suffix: str
          Suffix to append to the unit.

      Returns:
      --------
        : str
          Human-readable size.
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"

class AbsModule(torch.nn.Module, metaclass=ABCMeta):
  @abstract_attribute
  def labels(self) -> list|None:
    pass

  def __str__(self):
    """
      A verbose string representation of the neural network.

      Returns:
      --------
        : str
          labels, features, classifier
    """
    features_str = "\n  ".join(str(self.features).split("\n"))
    classifier_str = "\n  ".join(str(self.classifier).split("\n"))
    labels_str = textwrap.fill(str(self.labels)[1:-1], width=80)
    labels_str = labels_str.replace('\n', '\n' + ' '*4)
    formated_size = sizeof_formated(sys.getsizeof(self))
    meta_str = f"size: {formated_size},\tdropout: {self.dropout},\tinput size:{self.input_size}"

    training_history_str = ""
    for line in self.training_history:
      training_history_str += f"Epochs: {line[0][1]}, "
      training_history_str += f"Batch: {line[1][1]}, "
      training_history_str += f"Items: ({line[2][1]}, {line[2][2]})\n"

      time_f = time.strftime("%H:%M:%S", time.gmtime(line[3][1]))
      training_history_str += ' '*4 + f"Time: {time_f}, "
      training_history_str += f"Accuracy: {line[4][1]:.2f}%\n"

    return f"{self.__repr__()} (\n" + \
      f"  (meta): \n    {meta_str}\n" + \
      f"  (labels): \n    {labels_str}\n" + \
      f"  (features): {features_str}\n" + \
      f"  (classifier): {classifier_str}\n" + \
      f"  (training history): \n    {training_history_str[:-1]}\n" + \
    f")"

  @abstractmethod
  def create(self):
    pass

  @abstractmethod
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    pass

  @abstractmethod
  def parameters(self) -> list:
    pass

  def save(self, path: str) -> None:
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
    torch.save({
      'name': 'Basic',
      'dropout': self.dropout,
      'input_size': self.input_size,
      'labels': self.labels,
      'features': self.features,
      'classifier': self.classifier,
      'state_dict': self.state_dict(),
      'training_history': self.training_history,
    }, path)

  def load(self, path: str) -> None:
    """
      Parameters:
      -----------
        path: str
          path to load model

      Returns:
      --------
        None
    """
    data = torch.load(path)
    self.dropout = data['dropout']
    self.input_size = data['input_size']
    self.labels = data['labels']
    self.features = data['features']
    self.classifier = data['classifier']
    self.load_state_dict(data['state_dict'])
    self.training_history = data['training_history']

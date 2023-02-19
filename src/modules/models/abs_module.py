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

  def __str__(self):
    """
      A verbose string representation of the neural network.

      Returns:
      --------
        : str
          labels, features, classifier
    """
    features_str = "\n  ".join(str(self.features).split("\n"))
    labels_str = textwrap.fill(str(self.labels)[1:-1], width=80)
    labels_str = labels_str.replace('\n', '\n' + ' '*4)
    formated_size = sizeof_formated(sys.getsizeof(self))

    if hasattr(self, "classifier"):
      classifier_str = "\n  ".join(str(self.classifier).split("\n"))

    if hasattr(self, 'dropout'):
      meta_str = f"size: {formated_size},\tdropout: {self.dropout},\tinput size:{self.input_size}" + \
        f"\t label count (output): {self.num_classes}"
    else:
      meta_str = f"size: {formated_size},\tinput size:{self.input_size}," + \
        f"\t label count (output): {self.num_classes}"

    training_history_str = ""*4
    for line in self.training_history:
      training_history_str += f"Epochs: {line[0][1]}, "
      training_history_str += f"Batch: {line[1][1]}, "
      training_history_str += f"Items: ({line[2][1]}, {line[2][2]})\n"

      time_f = time.strftime("%H:%M:%S", time.gmtime(line[3][1]))
      training_history_str += ' '*6 + f"Accuracy: {line[4][1]:.2f}%, "
      training_history_str += f"Time: {time_f}\n" + " "*4
    training_history_str = training_history_str[:-(4-1)] # 4 will be a constant

    return f"{self.__repr__()} (\n" + \
      f"  (meta): \n    {meta_str}\n" + \
      f"  (labels): \n    {labels_str}\n" + \
      f"  (features): {features_str}\n" + \
      (f"  (classifier): {classifier_str}\n" if hasattr(self, 'classifier') else '') + \
      (f"  (training history): \n    {training_history_str[:-1]}" if len(self.training_history) > 0 else '') + \
    f")"

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
    obj = {
      'name': self.__repr__()[:-2], # Trim the () ex. Basic()/Basic
      'input_size': self.input_size,
      'labels': self.labels,
      'features': self.features,
      'state_dict': self.state_dict(),
    }

    if hasattr(self, 'training_history'):
      obj['training_history'] = self.training_history
    if hasattr(self, 'dropout'):
      obj['dropout'] = self.dropout
    if hasattr(self, 'classifier'):
      obj['classifier'] = self.classifier

    torch.save(obj, path)

  def load(self, state_dict: dict) -> None:
    """
      Parameters:
      -----------
        path: str
          path to load model

      Returns:
      --------
        None
    """
    self.load_state_dict(state_dict)

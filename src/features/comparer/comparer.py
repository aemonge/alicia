from dependencies.core import contextlib, io
from dependencies.datatypes import AbsModule, Type
from libs import PrettifyComparer
from features import Trainer

class Comparer(PrettifyComparer):
  """
    Compare several neural networks performances.
    Test their accuracy or their training and validation loss.

    Methods:
    --------

  """

  def __init__(self, Trainer: Type[Trainer], models: list[AbsModule], *, names: list[str]):
    """
      Initialize

      Parameters
      ----------
        Trainer : Type[Trainer]
          Class of the trainer to use.
        models : list[AbsModule]
          List of neural networks to compare.
        names : list[str]
          Names of the models to compare.

      Returns:
      --------
        None
    """
    self.Trainer = Trainer
    self.models = models
    self.names = names

  def accuracy(self, transform, *args, **kwargs) -> None:
    """
      Compute the overall test accuracy of the models and compare them.

      Parameters:
      -----------
        transform : Type[Transform]
          Class of the transform to use.
        args:
          Arguments to pass to the `test` method of each model.
        kargs:
          Keyword Arguments to pass to the `train` method of each model.

      Returns:
      --------
        None
    """
    self._loading()

    results = []
    for model in self.models:
      t = self.Trainer(model, transform)
      with contextlib.redirect_stdout(io.StringIO()) as f:
        t.test(*args, **kwargs)
        results.append(f.getvalue())

    self._print_results(results)

  def training(self, data_dir, transform, labels, batch_size, **kwargs) -> None:
    """
      Start parallel training of the models and compare their training and validation loss, time and accuracy.

      Parameters:
      -----------
        data_dir : str
          Path to the directory containing the images in /valid and /train
        transform : Type[Transform]
          Class of the transform to use.
        labels : str
          Path to the labels file
        batch_size : int
          Batch size
        kargs:
          Keyword Arguments to pass to the `Trainer()` of each model.

      Returns:
      --------
        None
    """
    self._loading()

    results = []
    try:
      for model in self.models:
        t = self.Trainer(model, transform, **kwargs)
        with contextlib.redirect_stdout(io.StringIO()) as f:
          t.train(data_dir, labels, batch_size, 1)
          results.append(f.getvalue())
    except Exception as e:
      self._terminate_loading()
      raise e

    self._print_results(results)

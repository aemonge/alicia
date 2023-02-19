from dependencies.fancy import colored, spinner
from dependencies.core import multiprocessing, time, abstractmethod, ABCMeta
from libs import colorize_diff

class PrettifyComparer(metaclass=ABCMeta):
  """
    Prettify the comparer by adding spinner, and nice diff output.
  """
  BAR_LENGTH = 55
  LOADING_ICONS = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
  WIDTH = 50

  @abstractmethod
  def __init__(self):
    """
      Implementing this __init__ **only** for typing purposes. The child class, shall
      not call this method directly.
    """
    self.models: list
    self.names: list[str]

  def __str__(self) -> str:
    """
      Return a string representation of the comparer.
    """
    last = len(self.models) - 2
    result = ""
    for i, _ in enumerate(self.models[:-1]):
      result += colorize_diff(self.models[i], self.models[i+1],
                              headers=(self.names[i], self.names[i+1]), omit_changes=True
                              )
      if i < last:
        result += '\n '+ colored('-' * self.WIDTH, 'black') +'\n\n'

    return  result

  def _terminate_loading(self) -> None:
    """
      Terminate the loading spinner. Made a method to be called in the child class.
    """
    print(f"\r{' ':>55}\r", end='')
    self.__loading.terminate();

  def _print_results(self, results: list[str]) -> None:
    """
      Print the results and compare them. This will also terminate the spinner.
    """
    self._terminate_loading()
    # print(results)
    last = len(self.models) - 2
    for i, _ in enumerate(self.models[:-1]):
      result = colorize_diff(results[i], results[i+1],
                             headers=(self.names[i], self.names[i+1]), omit_changes=True
                             )
      if i < last:
        result += '\n '+ colored('-' * self.WIDTH, 'black') +'\n\n'
      print(result)

  def __infinite_spinner(self) -> None:
    """
      Print a loading spinner.
    """
    s = spinner(icons=[' ' + colored(i, 'yellow')* self.WIDTH for i in self.LOADING_ICONS])
    while True:
      next(s)
      time.sleep(0.1)

  def _loading(self) -> None:
    """
      Start parallel loading spinner.
    """
    self.__loading = multiprocessing.Process(target=self.__infinite_spinner)
    self.__loading.start()

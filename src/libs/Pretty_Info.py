
# Globals
from matplotlib import pyplot as plt
# import plotext as plt
import numpy as np
import time

# Fancy
from loading_display import spinner
from loading_display import loading_bar
from termcolor import colored

class PrettyInfo:
  BAR_LENGTH = 55
  LOADING_ICONS = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']

  def __init__(self):
    self._spinners_ = {
      'train': spinner(icons=[' ' + colored(i, 'blue') for i in self.LOADING_ICONS]),
      'test':  spinner(icons=[' ' + colored(i, 'yellow') for i in self.LOADING_ICONS]),
      'valid': spinner(icons=[' ' + colored(i, 'green') for i in self.LOADING_ICONS])
    }

  def __get_step_color__(self, step: str = 'train') -> str:
    color = ""
    match step:
      case 'train':
        color = 'blue'
      case 'valid':
        color = 'green'
      case 'test':
        color = 'yellow'
    return color

  def _loading(self, ix, total, step: str = 'train') -> None:
    color = self.__get_step_color__(step)
    loading_bar(
      ix, total=total, bar_length=self.BAR_LENGTH, show_percentage=True, icon=colored('⠿', color)
    )

  def _spin(self, step: str = 'train') -> None:
    next(self._spinners_[step])

  def __backspace__(self, hard: bool = False):
    if hard:
      print('\033[F\033[K', end='') # back prev line and clear
    else:
      print('\r', end='\r')

  def _print_train_header(self, epochs, batch_size, train_loader_count, validate_loader_count) -> None:
    momentum_str = ""
    if hasattr(self, 'momentum') and self.momentum is not None:
      momentum_str = f",\tMomentum: {self.momentum}"

    print(f" Epochs: {epochs},\tBatch Size: {batch_size},\tLearning rate: {self.learning_rate}{momentum_str}\n",
          f"Items: [training: \"{train_loader_count:,}\" ,\tvalidation: \"{validate_loader_count:,}\"]\n")

  def _print_test_header(self, batch_size, count) -> None:
    print(f" Batch Size: {batch_size},\tItems : {count}\n")

  def _print_step_header(self, epochs, epoch) -> None:
    print(f"   Epoch: {epoch + 1}/{epochs} ({colored('traning', 'blue')} and {colored('validating', 'green')})")

  def _print_step(self, epoch, epochs, start_time, time_count,
                 tr_loss, vd_loss, vd_correct,
                 validate_loader_count, train_loader_count) -> float:

    training_loss = tr_loss / train_loader_count
    validate_loss = vd_loss / validate_loader_count

    time_now = time.time()
    time_count += time_now - start_time
    time_f = time.strftime("%H:%M:%S", time.gmtime(time_now - start_time))
    acc = vd_correct * 100 / validate_loader_count
    accuracy_f = colored(f"{acc:.4f}%", ('blue' if acc > 90 else ('green' if acc > 70 else 'red')))

    print(f"   Epoch: {epoch + 1}/{epochs}\n",
          f"     Losses [ training: {training_loss:.6f}, validate: {validate_loss:.6f} ]{' ':>20}\n",
          f"     Accuracy: {accuracy_f}, Time: {time_f}{' ':>30}\n",
    )

    return time_now

  def _print_total(self, correct, count, start_time):
    time_f = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    acc = (correct * 100 / count) #:.2f
    accuracy_f = colored(f"{acc:.3f}%", ('blue' if acc > 90 else ('green' if acc > 70 else 'red')))

    print()
    self.__backspace__(hard=True)
    print(f" Accuracy: {accuracy_f}, Time: {time_f}")


  def _print_t_step(self, start_time, t_correct, test_loader_count) -> None:
    time_f = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    acc = (t_correct * 100 / test_loader_count)
    accuracy_f = colored(f"{acc:.3f}%", ('blue' if acc > 90 else ('green' if acc > 70 else 'red')))

    self.__backspace__()
    print(f" Accuracy: {accuracy_f}, Time: {time_f}{' ':>40}")

  def _print_pbs(self, labels, probs, ax=None):
      probs = np.round(probs, 2)

      if ax is None:
          plt.subplot(1, 1)
          plt.barh(labels, probs)
      else:
          ax.barh(labels, probs)

  def _imshow(self, image, ax=None, title=None, tilted=False):
      """Imshow for Tensor."""
      if ax is None:
          _, ax = plt.subplots()

      if title is not None:
        if tilted:
          ax.set_ylabel(title.replace(' ', '\n'), fontsize=10)
          ax.yaxis.set_label_position("right")
          ax.tick_params(axis='y')
          ax.set_xticks([])
          ax.set_yticks([])
        else:
          ax.set_title(title)
          ax.axis("off")

      ax.imshow(image, interpolation='nearest')
      return ax

# Fancy,
from dependencies.core import time, glob, randrange, Image, os, np
from dependencies.fancy import spinner, loading_bar, colored, plt, plotext
from PIL.Image import Image as ImageDT
from matplotlib.axes import Axes

class PrettyTrain:
  """
    A parent class that prints the trainer data and analytics
      nicely and colored.

    Attributes
    ----------
      spinners_ : dict
        A dictionary of spinners, with different colors.

    Methods
    -------
      _loading(ix, total, step: str = 'train')
        Prints a loading bar.
      _spin(step: str)
        Prints a spinner.
      _print_train_header(self, epochs, batch_size, train_loader_count, validate_loader_count)
        Prints the training header.

  """
  BAR_LENGTH = 55
  LOADING_ICONS = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']

  def __init__(self):
    """
      Constructor. This is meant to be a parent Class
    """
    self.__enable_spinners = False # Loading bar seams enough
    if self.__enable_spinners:
      self._spinners_ = {
        'train': spinner(icons=[colored(i, 'blue') for i in self.LOADING_ICONS]),
        'test':  spinner(icons=[colored(i, 'yellow') for i in self.LOADING_ICONS]),
        'valid': spinner(icons=[colored(i, 'green') for i in self.LOADING_ICONS])
      }

  def __get_step_color__(self, step: str = 'train') -> str:
    """
      Gets the color of a step.

      Parameters
      ----------
        step : str
          The step to get the color of.

      Returns
      -------
        color : str
          The color of the step.
    """
    color = ""
    match step:
      case 'train':
        color = 'blue'
      case 'valid':
        color = 'green'
      case 'test':
        color = 'yellow'
    return color

  def _loading(self, ix: int, total: int, step: str = 'train') -> None:
    """
      Prints the loading bar.

      Parameters
      ----------
        ix: int
          The index of the loading bar.
        total: int
          The total number of items.
        step: str
          The step(color) of the loading bar.

      Returns:
      --------
        None
    """
    color = self.__get_step_color__(step)
    loading_bar(
      ix, total=total, label='  ', bar_length=self.BAR_LENGTH, show_percentage=True, icon=colored('⠿', color)
    )

  def _spin(self, step: str = 'train') -> None:
    """
      Prints a spinner.

      Parameters:
      -----------
        step: str
          The step(color) of the spinner.

      Returns:
      --------
        None
    """
    if self.__enable_spinners:
      next(self._spinners_[step])

  def __backspace__(self, hard: bool = False):
    """
      Prints a backspace character.

      Parameters:
      -----------
        hard: bool
          Whether to print a hard or soft backspace character.

      Returns:
      --------
        None
    """
    if hard:
      print('\033[F\033[K', end='') # back prev line and clear
    else:
      print('\r', end='\r')

  def _print_train_header(self, epochs:int, batch_size:int,
                          train_loader_count:int, validate_loader_count:int) -> None:
    """
      Prints the header of the training.

      Parameters:
      -----------
        epochs:int
          The number of epochs.
        batch_size:int
          The batch size.
        train_loader_count:int
          The number of train loader.
        validate_loader_count:int
          The number of validate loader.

      Returns:
      --------
        None

    """
    momentum_str = ""
    if hasattr(self, 'momentum') and self.momentum is not None:
      momentum_str = f",\tMomentum: {self.momentum}"

    print(f" Epochs: {epochs},\tBatch Size: {batch_size},\tLearning rate: {self.learning_rate}{momentum_str}\n",
          f"Items: [training: \"{train_loader_count:,}\" ,\tvalidation: \"{validate_loader_count:,}\"]\n")

  def _print_test_header(self, batch_size:int, count:int) -> None:
    """
      Prints the header of the testing.

      Parameters:
      -----------
        batch_size:int
          The batch size.
        count:int
          The total number of items.

      Returns:
      --------
        None

    """
    print(f" Batch Size: {batch_size},\tItems : {count}\n")

  def _print_step_header(self, epochs:int, epoch:int) -> None:
    """
      Prints the header of a step.

      Parameters:
      -----------
        epochs:int
          The number of epochs.
        epoch:int
          The epoch number.

      Returns:
      --------
        None
    """
    print(f"   Epoch: {epoch + 1}/{epochs} ({colored('traning', 'blue')} and {colored('validating', 'green')})")

  def _print_step(self, epoch:int, epochs:int, start_time:int, time_count:int,
                  tr_loss:float, vd_loss:float, vd_correct:int,
                  validate_loader_count:int, train_loader_count:int) -> float:
    """
      Prints a step.

      Parameters:
      -----------
        epoch : int
          The epoch number.
        epochs: int
          The number of epochs.
        start_time: int
          The start time.
        time_count: int
          The time count.
        tr_loss: float
          The training loss.
        vd_loss: float
          The validation loss.
        vd_correct: int
          The validation accuracy.
        validate_loader_count:int
          The number of validate loader.
        train_loader_count:int
          The number of train loader.

      Returns:
      --------
        : float
          The time passed from last step, used to track time
    """

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

  def _print_total(self, correct, count, start_time) -> None:
    """
      Prints the total.


      Parameters:
      -----------
        correct: int
          The number of correct items
        count: int
          The total number of items.
        start_time: int
          The start time.

      Returns:
      --------
        None
    """
    time_f = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    acc = (correct * 100 / count) #:.2f
    accuracy_f = colored(f"{acc:.3f}%", ('blue' if acc > 90 else ('green' if acc > 70 else 'red')))

    print(f" Accuracy: {accuracy_f}, Time: {time_f}")

  def _print_t_step(self, start_time, t_correct, test_loader_count) -> None:
    """
      Prints a test step.


      Parameters:
      -----------
        start_time: int
          The start time.
        t_correct: int
          The number of correct items.
        test_loader_count: int
          The number of test loader.

      Returns:
      --------
        None
    """
    time_f = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    acc = (t_correct * 100 / test_loader_count)
    accuracy_f = colored(f"{acc:.3f}%", ('blue' if acc > 90 else ('green' if acc > 70 else 'red')))

    self.__backspace__()
    print(f" Accuracy: {accuracy_f}, Time: {time_f}{' ':>40}")

  def _print_pbs_in_console(self, labels:dict, probs, subplot_idx) -> None:
    """
      Print the probability distribution in console.

      Parameters:
      -----------
        labels: dict
          The labels.
        probs:
          The probability distribution.
        ax: matplotlib.axes.Axes
          The axis to plot.

      Returns:
      --------
        None
    """
    probs = np.round(probs, 2)
    plotext.subplot(subplot_idx + 1, 1)
    plotext.bar(labels, probs, orientation = "h", width = 0.1)

  def _imshow_in_console(self, image:ImageDT, subplot_idx:int, *, title:str, tilted:bool =False)-> None:
    """
      Plot the image.

      Parameters:
      -----------
        image: Image
          The image to plot
        ax: matplotlib.axes.Axes
          The axis to plot.
        title: str
          The title of the Image.
        tilted: bool
          Whether the image title should be vertical.

      Returns:
      --------
        None
    """
    plotext.subplot(subplot_idx + 1, 2)
    plotext.image_plot(image)
    plotext.xlabel(title, xside=1)

  def _print_pbs(self, labels:dict, probs, ax:Axes) -> None:
    """
      Print the probability distribution.

      Parameters:
      -----------
        labels: dict
          The labels.
        probs:
          The probability distribution.
        ax: matplotlib.axes.Axes
          The axis to plot.

      Returns:
      --------
        None
    """
    probs = np.round(probs, 2)

    if ax is None:
      plt.subplot(1, 1)
      plt.barh(labels, probs)
    else:
      ax.barh(labels, probs)

  def _imshow(self, image:ImageDT, *, ax:Axes, title:str, tilted:bool =False)-> Axes:
    """
      Plot the image.

      Parameters:
      -----------
        image: Image
          The image to plot
        ax: matplotlib.axes.Axes
          The axis to plot.
        title: str
          The title of the Image.
        tilted: bool
          Whether the image title should be vertical.

      Returns:
      --------
        None
    """
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

  def show_test_results(self, path: str, labels: dict, count: int, tilted: bool = False) -> None:
    """
      Show the test results.

      Parameters:
      -----------
        path: str
          The path to the test images.
        labels: dict
          The class labels.
        count: int
          The total number of items do show.
        tilted: bool
          Whether the image title should be vertical.

      Returns:
      --------
        None
    """
    all_imgs = [
      file for file in glob.glob(f"{path}/**/*.@(jpg|jpeg|png)", flags=glob.EXTGLOB)
    ]

    _, ax = plt.subplots(count, 2, width_ratios=[4,1])
    if count == 1:
      ax = [ax]

    if tilted:
      plt.subplots_adjust(left=0.22, bottom=0.08, top=0.98, hspace=0.6, wspace=0)
    else:
      plt.subplots_adjust(left=0.22, bottom=0.08, top=0.90, hspace=0.6, wspace=0.35, right=0.86)

    for c_idx in range(count):
      ix = randrange(len(all_imgs))
      img_path = all_imgs[ix]

      image = Image.open(img_path)
      file_name = os.path.basename(img_path)

      probs, classes = self.predict(image, topk=3)
      self._print_pbs(classes, probs, ax=ax[c_idx][0])
      self._imshow(self.transforms['display'](image), title=labels[file_name], ax=ax[c_idx][1], tilted=tilted)

    plt.show()

  def show_test_results_in_console(self, path: str, labels: dict, count: int) -> None:
    """
      Show the test results in console.

      Parameters:
      -----------
        path: str
          The path to the test images.
        labels: dict
          The class labels.
        count: int
          The total number of items do show.
        tilted: bool
          Whether the image title should be vertical.

      Returns:
      --------
        None
    """
    all_imgs = [
      file for file in glob.glob(f"{path}/**/*.@(jpg|jpeg|png)", flags=glob.EXTGLOB)
    ]

    plotext.subplots(count, 2)
    plotext.plot_size(80, 15*count)

    for c_idx in range(count):
      ix = randrange(len(all_imgs))
      img_path = all_imgs[ix]

      image = Image.open(img_path)
      file_name = os.path.basename(img_path)

      probs, classes = self.predict(image, topk=3)
      self._print_pbs_in_console(classes, probs, c_idx)
      self._imshow_in_console(img_path, c_idx, title=labels[file_name])

    plotext.show()
    plotext.clf()

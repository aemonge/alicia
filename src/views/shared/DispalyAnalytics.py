import plotext as plt
import os
import time
from termcolor import colored


def print_t_step(start_time, t_correct, test_loader_count) -> None:
  time_f = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
  acc = (t_correct * 100 / test_loader_count)
  accuracy_f = colored(f"{acc:.3f}%", ('blue' if acc > 90 else ('green' if acc > 70 else 'red')))

  print(f" Accuracy: {accuracy_f}, Time: {time_f}{' ':>40}")

def print_step(epoch, epochs, start_time, time_count,
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

def print_total(correct, count, start_time):
  time_f = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
  acc = (correct * 100 / count) #:.2f
  accuracy_f = colored(f"{acc:.3f}%", ('blue' if acc > 90 else ('green' if acc > 70 else 'red')))

  print('\033[F\033[K', end='\r') # back prev line and clear
  print(f" Accuracy: {accuracy_f}, Time: {time_f}")

class DispalyAnalytics:
  """
  This class is responsible for displaying the results of the analysis of the AI models performance and accuracy.

    Attributes
    ----------
      width : int
        width of the display.

    Methods
    -------
      header(model="Dummy", gpu=False):
        Pretty print Header.
      footer(total_time=3, accuracy=0.98)
        Pretty print Footer with the result
      step(epoch=0, run_time=0.0, step_analysis={})
        Pretty print the current model step.
      graph(steps)
        Pretty print the current model graph.
  """

  LINE_CHAR = '—'
  PADDING = 1
  COLOR = 'green'
  PADDING_CHAR = '\t'
  BREAK_CHAR = '\n'

  def __init__(self, width=65, verbose=False):
    """
      Initializes the class.

      Parameters
      ----------
        width : int
          width of the display.
    """
    self.width = width
    self.text_adjustment = 8
    self.verbose = verbose

  def header(self, model="Dummy", gpu=False, verbose = {}):
    """
      Pretty print Header.

      Parameters
      ----------
        model : string
          model name.
        gpu : bool
          use gpu or not.

      Returns
      -------
        None
    """
    if not verbose:
      os.system('clear')

    print(self.__get_hr(color='blue', padding=0))
    gpu_msg = f"GPU { colored('enabled', attrs=['bold']) if gpu else 'disabled' }"
    print(f" Model \"{colored(model, 'blue')}\" Starts with {gpu_msg}")
    if verbose: # TODO: incorporate the transformation smartly, or simply skip them they should match
      print(self.__get_hr(char="―", color='grey', padding=0))
      print('  Classes: ', verbose['classes'].keys())
      print(self.__get_hr(char="―", color='grey', padding=0))
      print(' ', verbose['model'])
      print(self.__get_hr(char="―", color='grey', padding=0))

      print('  images: \t\t', verbose['images'].shape, '\t', verbose['images'].dtype)
      reimage= verbose['images'].view(verbose['images'].shape[0], -1)
      print('  images.transformed: \t', reimage.shape, '\t\t', reimage.dtype)
      print(self.__get_hr(char="―", color='grey', padding=0))

      print('  labels: \t\t', verbose['labels'].shape, '\t\t', verbose['labels'].dtype)

    print(self.__get_hr(char="― ", color='cyan', padding=0))

  def footer(self, total_time=3, accuracy=0.98):
    """
      Pretty print Footer with the result

      Parameters
      ----------
        total_time : int
          total time of the analysis.
        accuracy : float
          Final accuracy.

      Returns
      -------
        None
    """

    time_f = time.strftime("%H:%M:%S", time.gmtime(total_time))
    print(self.__get_hr(char= "― ", color='cyan', padding=0))
    msg = f" ∑ Accuracy {colored(str(round(accuracy, 3)), attrs=['bold'])}%"
    msg += f"{self.PADDING_CHAR}∑ Total Time: {colored(str(time_f), attrs=['bold'])}"
    print(msg)
    print(self.__get_hr(color='blue', padding=0))

  def step(self, epoch=0, run_time=0.0, step_analysis={}):
    """
      Pretty print the current model step.

      Parameters
      ----------
        epoch : int
          current epoch.
        run_time : float
          current run time.
        step_analysis : dict
          step analysis.

      Returns
      -------
        None
    """
    msg = self.__get_step_body(epoch, step_analysis)
    msg += self.__get_step_footer(epoch, run_time)
    print(msg)

  def graph(self, steps):
    """
      Pretty print the current model graph.

      Parameters
      ----------
        steps : list
          current steps.

      Returns
      -------
        None
    """
    plt.plot(steps['trainingAcc'], label = 'Training', color= 'blue')
    plt.plot(steps['testAcc'], label = 'Test', color= 'green')
    plt.plot_size(self.width, 30)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ticks_style('bold')
    plt.title('Overall Loss')
    print(self.__add_padding_to_graph(plt.build()))

  def test(self, fn, *args, **kwargs):
    """
      Pretty print Header for Test, and also uses `fn` to print inside the test box desired results

      Parameters
      ----------
      fn : function
        test function.
      *args : list
        the arguments for the fn function
      **kwargs : dict
        the keyword arguments for the fn function

      Returns
      -------
        None
    """
    print(self.__get_hr(color='green', padding=0))
    print(f" Tests: ")
    print(self.__get_hr(color='grey', padding=0))
    fn(*args, **kwargs)
    print(self.__get_hr(color='green', padding=0))

  def __get_hr(self, char = LINE_CHAR, color = COLOR, padding = PADDING):
    """
      Pretty print line as a separator.

      Parameters
      ----------
        char : string
          separator character.
        color : string
          color of the separator.
        padding : int
          padding of the separator.

      Returns
      -------
        None
    """
    if len(char) > 1:
      msg = char * (int(self.width / 2) + int(self.text_adjustment / 2) - (2 * padding))
      msg += char[0]
    else:
      msg = char * (self.width + self.text_adjustment - (self.text_adjustment * padding))

    return(self.PADDING_CHAR * padding + colored(msg, color))

  def __get_step_footer(self, epoch, run_time):
    """
      Pretty print the step footer.

      Parameters
      ----------
        epoch : int
          current epoch.
        run_time : float
          current run time.

      Returns
      -------
        string
          The step footer message, formatted.
    """
    msg = self.BREAK_CHAR
    msg += self.__get_hr(color='green')
    msg += colored(f"{self.BREAK_CHAR}{self.PADDING_CHAR}Epoch: {epoch}", attrs=["bold"])
    msg += f"{self.PADDING_CHAR}Learn: x"
    msg += f"{self.PADDING_CHAR}Time: {run_time:.5f}"
    msg += self.BREAK_CHAR
    msg += self.__get_hr(color='green')
    msg += self.BREAK_CHAR

    return msg

  def __get_step_body(self, epoch, step_analysis):
    """
      Pretty print the step body

      Parameters
      ----------
        epoch : int
          current epoch.
        step_analysis : dict
          step analysis.


      Returns
      -------
        string
          The step body message, formatted.
    """
    msg = ""
    for analytics_type, analytics in step_analysis.items():
      mean, std = analytics["mean"], analytics["std"]

      msg += colored(f"{self.PADDING_CHAR}{analytics_type.capitalize()} Accuracy:", attrs=["underline"])
      msg += self.BREAK_CHAR
      msg += f"{self.PADDING_CHAR}Mean: {mean:.5f}{self.PADDING_CHAR}"
      msg += f"{self.PADDING_CHAR}Std: {std}{self.PADDING_CHAR}"
      msg += f"{self.PADDING_CHAR}Loss: {-epoch}{self.BREAK_CHAR * 2}"

    return msg[:-2]

  def __add_padding_to_graph(self, graph):
    """
      Add padding to the graph.

      Parameters
      ----------
        graph : plotext.Graph
          current graph.

      Returns
      -------
        None
    """
    padding = lambda s: (f"{self.PADDING_CHAR}{s}")
    return self.BREAK_CHAR.join(list(map(padding, graph.split(self.BREAK_CHAR))))

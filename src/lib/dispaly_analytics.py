import plotext as plt
import os
from termcolor import colored

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
      footer(total_time=3, validation_accuracy=0.98, validation_loss=5)
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

  def __init__(self, width=55, verbose=False):
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
    if not self.verbose:
      os.system('clear')

    print(self.__get_hr(color='blue', padding=0))
    gpu_msg = f"GPU { colored('enabled', attrs=['bold']) if gpu else 'disabled' }"
    print(f"Model \"{colored(model, 'blue')}\" Starts with {gpu_msg}")
    if verbose: # TODO: incorporate the transformation smartly, or simply skip them they should match
      print(self.__get_hr(char="―", color='grey', padding=0))
      print(verbose['model'])
      print(self.__get_hr(char="―", color='grey', padding=0))
      print('  images.dtype: \t', verbose['images'].dtype)
      print('  images.shape: \t', verbose['images'].shape)
      print('  images.re-shaped: \t', verbose['images'].view(verbose['images'].shape[0], -1).shape)
      print(self.__get_hr(char="―", color='grey', padding=0))
      print('  labels.shape: \t', verbose['labels'].shape)
      print('  labels.re-shape: \t', verbose['labels'][:,0].shape)
      print('  labels.re-dtype: \t', verbose['labels'][:,0].long().dtype)

    print(self.__get_hr(char="― ", color='cyan', padding=0))

  def footer(self, total_time=3, validation_accuracy=0.98, validation_loss=5):
    """
      Pretty print Footer with the result

      Parameters
      ----------
        total_time : int
          total time of the analysis.
        validation_accuracy : float
          validation accuracy.
        validation_loss : int
          validation loss.

      Returns
      -------
        None
    """
    print(self.__get_hr(char= "― ", color='cyan', padding=0))
    msg = f"∑ Accuracy {colored(str(validation_accuracy), attrs=['bold'])}%"
    msg += f"{self.PADDING_CHAR}∑ Loss: {colored(str(validation_loss), attrs=['bold'])}"
    msg += f"{self.PADDING_CHAR}∑ Total Time: {colored(str(total_time), attrs=['bold'])}s"
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

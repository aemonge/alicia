import time
import random
import plotext as plt
from termcolor import colored

SLEEP_TIME = 0.02
WIDTH = 65
N_EPOCHS = 10
now = time.time

def print_header(model="Dummy", gpu=False):
  """
    Pretty print Header
  """
  print(hr_line(color='blue', padding=0))
  gpu_msg = f"GPU { colored('enabled', attrs=['bold']) if gpu else 'disabled' }"
  print(f"Model \"{colored(model, 'blue')}\" Starts with {gpu_msg}")
  print(hr_line(char="― ", color='cyan', padding=0))

def hr_line(char = "—", color = "green", padding = 1):
  text_adjustment = 8
  padds = "\t"*padding
  if len(char) > 1:
    msg = char * (int(WIDTH/2) + int(text_adjustment/2) - (2*padding))
    msg += char[0]
  else:
    msg = char * (WIDTH + text_adjustment - 8*padding)

  return(padds + colored(msg, color))

def print_footer(total_time=3, validation_accuracy=0.98, validation_loss=5):
  """
    Pretty print Footer with the result
    """
  print(hr_line(char= "― ", color='cyan', padding=0))
  msg = f"∑ Accuracy {colored(str(validation_accuracy), attrs=['bold'])}%"
  msg += f"\t∑ Loss: {colored(str(validation_loss), attrs=['bold'])}"
  msg += f"\t∑ Total Time: {colored(str(total_time), attrs=['bold'])}s"
  print(msg)
  print(hr_line(color='blue', padding=0))


def print_step_footer(epoch, run_time):
  """
    Pretty print the step header
  """
  msg = "\n"
  msg += hr_line(color='green')
  msg += colored(f"\n\tEpoch: {epoch}", attrs=["bold"])
  msg += "\tLearn: x"
  msg += f"\tTime: {run_time:.5f}"
  msg += "\n"
  msg += hr_line(color='green')
  msg += "\n"

  return msg


def print_step_body(epoch, step_analysis):
  """
    Pretty print the step body
  """
  msg = ""
  for analytics_type, analytics in step_analysis.items():
    mean, std = analytics["mean"], analytics["std"]

    msg += colored(f"\t{analytics_type.capitalize()} Accuracy:", attrs=["underline"]) + "\n"
    msg += f"\tMean: {mean:.5f}\t"
    msg += f"\tStd: {std}\t"
    msg += f"\tLoss: {-epoch}\n\n"

  return msg[:-2]


def m_print(epoch=0, run_time=0.0, step_analysis={}):
  """
    Pretty print
    """
  msg = ""

  msg += print_step_body(epoch, step_analysis)
  msg += print_step_footer(epoch, run_time)

  # step_analysis.
  print(msg)

def add_padding(graph):
  padding = lambda s: (f"\t{s}")
  return "\n".join(list(map(padding, graph.split('\n'))))

def graph(steps):
  plt.plot(steps['trainingAcc'], label = 'Training', color= 'blue')
  plt.plot(steps['testAcc'], label = 'Test', color= 'green')
  plt.plot_size(WIDTH, 30)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.ticks_style('bold')
  plt.title('Overall Loss')
  print(add_padding(plt.build()))

def dummy(show_graph=True, gpu=False):
  """
    Simply a dummy for creating the print and charts to check further models performance.
    """
  timer = time.time()
  steps = {
    "trainingAcc": [],
    "testAcc": []
  }
  print_header(gpu=gpu)

  for epoch in range(N_EPOCHS):
    time.sleep(SLEEP_TIME)

    analytics = {
      "test": {
        "mean": random.random(),
        "std": random.randint(0, 5),
        "loss": random.randint(0, 10 - epoch)
      },
      "training": {
        "mean": random.random(),
        "std": random.randint(0, 5),
        "loss": random.randint(0, 10 - epoch)
      },
    }
    steps["testAcc"].append(analytics["test"]["loss"])
    steps["trainingAcc"].append(analytics["training"]["loss"])
    m_print(epoch=epoch, run_time=(now() - timer), step_analysis=analytics)
    timer = time.time()

  if (show_graph):
    graph(steps)

  print_footer()

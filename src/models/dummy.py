import time
import random

class DummyModel:
  def __init__(self, epochs = 2, show_graph=True, gpu=False, sleep_time=0.02, step_print_fn=None):
    """
      A Dummy Model that fakes all the data with some milliseconds. Useful for debugging.

      Attributes
      ----------
        epochs : int
          number of epochs to train
        show_graph : bool
          whether to show the graph
        gpu : bool
          whether to use gpu
        sleep_time : float
          sleep time between epochs
        step_print_fn : function
          function to print the training progress

      Methods
      -------
    """
    self.epochs = epochs
    self.show_graph = show_graph
    self.gpu = gpu
    self.print = step_print_fn
    self.now = time.time
    self.sleep_time = sleep_time
    self.steps = {
      "trainingAcc": [],
      "testAcc": []
    }

  def run(self):
    """
      Run the dummy model.

      Parameters
      ----------
        None

      Returns
      -------
        None
    """
    timer = time.time()
    self.print.header(gpu=self.gpu)

    for epoch in range(self.epochs):
      time.sleep(self.sleep_time)

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
      self.steps["testAcc"].append(analytics["test"]["loss"])
      self.steps["trainingAcc"].append(analytics["training"]["loss"])
      self.print.step(epoch=epoch, run_time=(self.now() - timer), step_analysis=analytics)
      timer = time.time()

    if (self.show_graph):
      self.print.graph(self.steps)

    self.print.footer()

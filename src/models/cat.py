import plotext as plt

class Cat:
  def __init__(self, data_dir = './data'):
    """
      Display a cat image

      Parameters
      ----------
      data_dir : str, optional
        The path where the `cat.jpg` image is stored
    """
    self.data_dir = data_dir

  def run(self):
    """
      Show the Cat

      Returns
      -------
        None
    """
    plt.image_plot(f'{self.data_dir}/cat.jpg')
    plt.show()

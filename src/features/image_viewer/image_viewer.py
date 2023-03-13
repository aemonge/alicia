from ast import Try
from dependencies.core import os, Image, random
from libs import ImageToMatPlotLib
from dependencies.fancy import plt
from dependencies.datatypes import *

class ImageViewer(object):
  """
    A Class to display a random set of images, form a chosen directory, using matplotlib.
    It will display the before and after images, by applying the transformation to the images.
  """

  def __init__(self, image_dir:str, transform =None) -> None:
    """
      Constructor.

      Parameters:
      -----------
        image_dir:
          The directory containing the images.
        transform:
          The transformation to apply to the images.
    """
    self.image_dir = image_dir
    self.transform = transform

  def __call__(self, count:int = 10) -> None:
    """
      Display a random set of images, form a chosen directory, using matplotlib.

      Parameters:
      -----------
        count:
          The number of images to display.
    """
    self.images = self.load_images(count)

    _, axs = plt.subplots(count, 2)
    if count == 1:
      axs = [axs]

    for i in range(count):
      axs[i][0].plot()
      axs[i][0].imshow(self.images[i], )
      axs[i][0].axis('off')

      axs[i][1].plot()
      transformed_image = self.transform(self.images[i])

      if not isinstance(transformed_image, ndarray):
        transformed_image = ImageToMatPlotLib((-1,))(transformed_image)

      axs[i][1].imshow(transformed_image)
      axs[i][1].axis('off')
    plt.show()

  def load_images(self, count:int = 10) -> List[ImageDT]:
    """
      Load the random images from the chosen directory.

      Parameters:
      -----------
        count:
          The number of images to display.

      Returns:
      --------
        A list of images.
    """
    images = []
    list_image_paths = os.listdir(self.image_dir)
    random.shuffle(list_image_paths)

    for filename in list_image_paths[:count]:
      image = plt.imread(os.path.join(self.image_dir, filename))
      image = Image.fromarray(image)
      images.append(image)
    return images

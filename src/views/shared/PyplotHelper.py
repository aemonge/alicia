import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms as Transforms

def imshow(image, ax=None, title=None, tilted=False):
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

def print_pbs(labels, probs, ax=None):
    probs = np.round(probs, 2)

    if ax is None:
        plt.subplot(1, 1)
        plt.barh(labels, probs)
    else:
        ax.barh(labels, probs)

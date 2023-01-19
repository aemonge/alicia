import click

from lib.torchvision_downloader import TorchvisionDownloader

DATASETS=['MNIST', 'FashionMNIST']

@click.command()
@click.pass_context
@click.argument("dataset", default='MNIST', type=click.Choice(DATASETS))
@click.argument("target_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
def download(ctx, dataset, target_dir):
  """
    Download a MNIST dataset with PyTorch.

    The downloaded files will be process out to leave the raw JPGs images in the target directory,
      with a `labels.csv` file.
  """
  if dataset == 'MNIST':
    TorchvisionDownloader(image_dir = target_dir, dataset = 'MNIST').call()
  elif dataset == 'FashionMNIST':
    TorchvisionDownloader(image_dir = target_dir, dataset = 'FashionMNIST').call()
  else:
    print('Not implemented, yet 🐼')

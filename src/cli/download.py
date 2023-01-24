import click

from lib.torchvision_downloader import TorchvisionDownloader

DATASETS=['MNIST', 'FashionMNIST']

@click.command()
@click.pass_context
@click.argument("dataset", default='MNIST', type=click.Choice(DATASETS))
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.option("-s", "--split-percentage", default=70, type=click.INT)
def download(ctx, dataset, data_dir, split_percentage):
  """
    Download a MNIST dataset with PyTorch and split it into `/test` `/train` directories.

    The downloaded files will be process out to leave the raw JPGs images in the both directories
      with a `labels.csv` file on them.
  """
  TorchvisionDownloader(
    dir = data_dir, dataset = dataset, split_percentage = split_percentage
  ).call()

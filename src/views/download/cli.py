import click

from features.torchvision_downloader.TorchvisionDownloader import TorchvisionDownloader

DATASETS=['MNIST', 'FashionMNIST']

@click.command()
@click.pass_context
@click.argument("dataset", default='MNIST', type=click.Choice(DATASETS))
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.option("-s", "--split-percentage", default=(0.65, 0.25, 0.1), type=(float, float, float),
  help='The split percentage triplet "-s 0.65 0.25 0.1"'
)
def download(ctx, dataset, data_dir, split_percentage):
  """
    Download a MNIST dataset with PyTorch and split it into `./train`, `./valid`, and `./test` directories.

    The download process will also generate the `./labels.csv` file containing the labels of all sets.
  """
  TorchvisionDownloader(data_dir, dataset, split_percentage).call()

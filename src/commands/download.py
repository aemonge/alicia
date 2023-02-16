
from dependencies.core import click, torchvision
from features import TorchvisionDownloader

DATASETS=torchvision.datasets.__all__

@click.command()
@click.pass_context
@click.argument("dataset", default='MNIST', type=click.Choice(DATASETS))
@click.argument("data_dir", type=click.Path(dir_okay=True, readable=True), required=1)
@click.option("-s", "--split-percentage", default=(0.65, 0.25, 0.1), type=(float, float, float),
  help='The split percentage triplet "-s 0.65 0.25 0.1"'
)
@click.option("-f", "--force", type=click.BOOL, default=False, is_flag=True)
def download(_, dataset, data_dir, split_percentage, force):
  """
    Download a MNIST dataset with PyTorch and split it into `./train`, `./valid`, and `./test` directories.

    The download process will also generate the `./labels.csv` file containing the labels of all sets.
  """
  TorchvisionDownloader(data_dir, dataset, split_percentage).call(force)

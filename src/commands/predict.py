from dependencies.core import click, torch
from modules import models
from modules.transforms.image_transforms import ImageTransforms
from features import Trainer

@click.command()
@click.pass_context
@click.argument("model_file", type=click.Path(file_okay=True, exists=True, readable=True), required=1)
@click.argument("image", type=click.Path(exists=True, file_okay=True, readable=True), required=1)
@click.option("-k", "--top-k", default=1, type=click.INT, help="Show the top-k most likely categories.")
def predict(_, model_file, image, top_k):
  """
    Predict images using a pre trained model, for a given folder and with a categories file.
  """
  data = torch.load(model_file)
  model = getattr(models, data['name'])(**{"data": data})

  trainer = Trainer(model, ImageTransforms)
  probs, labels = trainer.predict_image(image, topk= top_k)

  if top_k == 1:
    print(f"  \"{labels[0]}\"")
  else:
    for i in range(top_k):
      print(f"  {labels[i]}:\t{probs[i]:04.1f}%")

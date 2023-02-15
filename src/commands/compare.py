from dependencies.core import click
from commands.compare_info import diff_info
from commands.compare_step_speed import step_speed
from commands.compare_accuracy import accuracy

@click.group()
@click.pass_context
def compare(_):
  """
    Compare the info, accuracy, and step speed two (or more by pairs) trained models on a test set.
  """
  pass

compare.add_command(diff_info)
compare.add_command(accuracy)
compare.add_command(step_speed)

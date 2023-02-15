from dependencies.core import difflib
from dependencies.fancy import colored

def colorize_diff(a, b, *, headers: tuple[str,str], omit_changes: bool = False) -> str:
  """
    Colorize the differences between two strings.

    Parameters:
    -----------
      a : str
        The first string.
      b : str
        The second string.
      headers : tuple[str, str]
        The headers to use as title for the diff

    Returns:
    --------
      : str
        The colorized diff.
  """
  diffs = difflib.ndiff(str(a).splitlines(keepends=True), str(b).splitlines(keepends=True))

  output = ""
  if headers:
    output = f"  {colored(headers[0], 'blue')}, {colored(headers[1], 'green')}\n\n"

  for line in diffs:
    if line.startswith('+'):
      output += '  ' + colored(line[2:], 'green')
    elif line.startswith('-'):
      output += '  ' + colored(line[2:], 'blue')
    elif line.startswith('?'):
      if not omit_changes:
        output += '  ' + colored(line[2:], 'yellow')
    else:
      output += line

  return output

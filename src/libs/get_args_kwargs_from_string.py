from dependencies.core import ast

def get_args_kwargs_from_string(string) -> tuple[str,list,dict]:
  """
    Parse a function call string and return the function name, the
    positional arguments and the keyword arguments.

    Returns:
    --------
      tuple(str,list,dict)
        0: func_name. the name of the function
        1: args. a list of positional
        2: kwargs. a dictionary of keyword arguments
  """
  args = []
  kwargs = {}

  # Split the string into the function name and argument string
  parts = string.strip().split("(")
  if len(parts) != 2 or not parts[1].endswith(")"):
    raise ValueError(f"Invalid function name: {parts[0]}")

  func_name = parts[0]
  arg_str = parts[1][:-1]

  # Parse the argument string using ast.parse()
  try:
    parsed = ast.parse(f"dummy({arg_str})", mode="eval")
  except SyntaxError:
    raise ValueError(f"Invalid function arguments: {arg_str}")

  # Extract the positional arguments
  for arg_node in parsed.body.args: # pyright: reportGeneralTypeIssues=false
    try:
      arg_value = ast.literal_eval(arg_node)
    except (ValueError, SyntaxError):
      raise ValueError(f"Invalid function arguments: {arg_node}")
    args.append(arg_value)

  # Extract the keyword arguments
  for kwarg_node in parsed.body.keywords: # pyright: reportGeneralTypeIssues=false
    key = kwarg_node.arg
    try:
      value = ast.literal_eval(kwarg_node.value)
    except (ValueError, SyntaxError):
      raise ValueError(f"Invalid function arguments: {kwarg_node}")
    kwargs[key] = value

  return func_name, args, kwargs

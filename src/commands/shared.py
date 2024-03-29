from dependencies.core import csv

def labels_reader(file: str, _sorted: bool = True) -> list|dict:
  """
    Reads labels from a CSV file.

    Parameters:
    -----------
      file: str
        Path to the CSV file.
      _sorted: bool
        Whether the labels should be sorted and returned as a list.

    Returns:
    --------
      list|dict
        Labels in the CSV file as a list or dictionary, dependent on the `sorted` parameter
  """
  labels = set() if _sorted else {}
  with open(file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    if _sorted:
      for _, label in reader:
        labels.add(label)
    else:
      for filename, label in reader:
        labels[filename] = label

  return sorted(list(labels)) if _sorted else labels

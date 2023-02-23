def sizeof_formated(num, suffix='B'):
    """
      Converts a size in bytes to a human-readable format.

      Parameters:
      -----------
        num: int
          Size in bytes.
        suffix: str
          Suffix to append to the unit.

      Returns:
      --------
        : str
          Human-readable size.
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0

import pytest
from src.libs import sizeof_formated

@pytest.fixture
def sizes_fixture():
  return [
    ("23.0", "23.0 B"),
    ("1023.0", "1023.0 B"),
    ("1023.5", "1023.5 B"),
    ("1024.0", "1.0 KiB"),
    ("2048.0", "2.0 KiB"),
    ("2560.0", "2.5 KiB"),
    ("1048576.0", "1.0 MiB"),
    ("1073741824.0", "1.0 GiB"),
    ("1099511627776.0", "1.0 TiB"),
    ("1125899906842624.0", "1.0 PiB"),
    ("1.180591620717411e21", "1024.0 EiB"),
    ("1.180591620717411e22", "10.0 ZiB"),
  ]

class Test_sizeof_formated:
  def should_return_in_the_right_format(self, sizes_fixture):
    for sizes in sizes_fixture:
      assert sizeof_formated(float(sizes[0])) == sizes[1]

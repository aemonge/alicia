import pytest
from src.libs.get_args_kwargs_from_string import get_args_kwargs_from_string

@pytest.fixture
def with_args_only_fixture():
  return 'hello(\'world\', 2)'

@pytest.fixture
def with_kwargs_only_fixture():
  return 'hello(foo=\'world\', bar=2)'

@pytest.fixture
def with_both_kwargs_and_args_fixture():
  return 'hello(1, True, foo=\'world\', bar=2)'

@pytest.fixture
def with_no_args_nor_kwargs_fixture():
  return 'hello()'

@pytest.fixture
def with_wrong_args_fixture():
  return 'hello(true)'

@pytest.fixture
def with_wrong_kwargs_fixture():
  return 'hello(foo=bar)'

@pytest.fixture
def wrong_fixture():
  return 'hello('

@pytest.fixture
def wrong_syntax_fixture():
  return 'hello("foo":"bar")'

@pytest.fixture
def wrong_syntax_and_variables_fixture():
  return 'hello(foo:bar)'

class Test_get_args_kwargs_from_string:
  def should_get_the_name_of_the_function(self, with_no_args_nor_kwargs_fixture):
    f_name, _, __ = get_args_kwargs_from_string(with_no_args_nor_kwargs_fixture)
    assert f_name == 'hello'

  def should_get_the_name_of_the_function_with_no_args_nor_kwargs(self, with_no_args_nor_kwargs_fixture):
    f_name, f_args, f_kwargs = get_args_kwargs_from_string(with_no_args_nor_kwargs_fixture)
    assert f_name == 'hello'
    assert f_args == []
    assert f_kwargs == {}

  def should_get_the_name_of_the_function_with_args_but_no_kwargs(self, with_args_only_fixture):
    f_name, f_args, f_kwargs = get_args_kwargs_from_string(with_args_only_fixture)
    assert f_name == 'hello'
    assert f_args == ['world', 2]
    assert f_kwargs == {}

  def should_get_the_name_of_the_function_with_no_args_but_with_kwargs(self, with_kwargs_only_fixture):
    f_name, f_args, f_kwargs = get_args_kwargs_from_string(with_kwargs_only_fixture)
    assert f_name == 'hello'
    assert f_args == []
    assert f_kwargs == {"foo":"world", "bar":2}

  def should_get_the_name_of_the_function_with_args_and_kwargs(self, with_both_kwargs_and_args_fixture):
    f_name, f_args, f_kwargs = get_args_kwargs_from_string(with_both_kwargs_and_args_fixture)
    assert f_name == 'hello'
    assert f_args == [1, True]
    assert f_kwargs == {"foo":"world", "bar":2}

  def should_fails_to_func_name(self, wrong_fixture):
    with pytest.raises(ValueError):
      get_args_kwargs_from_string(wrong_fixture)

  def should_fails_to_parse_args(self, with_wrong_args_fixture):
    with pytest.raises(ValueError):
      get_args_kwargs_from_string(with_wrong_args_fixture)

  def should_fails_to_parse_kwargs(self, with_wrong_kwargs_fixture):
    with pytest.raises(ValueError):
      get_args_kwargs_from_string(with_wrong_kwargs_fixture)

  def should_fails_to_parse_kwargs_with_wrong_syntax_and_variables(self, wrong_syntax_and_variables_fixture):
    with pytest.raises(ValueError):
      get_args_kwargs_from_string(wrong_syntax_and_variables_fixture)

  def should_fails_to_parse_kwargs_with_wrong_syntax(self, wrong_syntax_fixture):
    with pytest.raises(ValueError):
      get_args_kwargs_from_string(wrong_syntax_fixture)

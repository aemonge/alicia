tox -e build
tox -av

pip install -e .
aeimg-classify 10

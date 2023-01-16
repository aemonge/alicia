tox -e build
tox -av

pip install -e .
aeimg-classify 10
aeimg-classify data
ls src/aeimg_classifier/* src/lib/* | entr -rc aeimg-classify data
ls src/aeimg_classifier/* src/lib/* | entr -rc aeimg-classify -a basic data/mnist-original.mat
ls src/aeimg_classifier/* src/lib/* src/models/* | entr -rc aeimg-classify -a basic data/mnist-original.mat
ls src/aeimg_classifier/* src/lib/* src/models/* | entr -rc aeimg-classify -a basic data/mnist-numbers
ls src/aeimg_classifier/* src/lib/* src/models/* | entr -rc aeimg-classify -a basic data/mnist-numbers-tiny
ls src/aeimg_classifier/* src/lib/* src/models/* | entr -rc aeimg-classify -a basic data/mnist-numbers-tiny --verbose

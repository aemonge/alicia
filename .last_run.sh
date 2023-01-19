tox -e build
tox -av

pip install -e .
aeimg-classify 10
aeimg-classify data
ls src/*/* | entr -rc aeimg-classify data
ls src/*/* | entr -rc aeimg-classify -a basic data/mnist-original.mat
ls src/*/* | entr -rc aeimg-classify -a basic data/mnist-original.mat
ls src/*/* | entr -rc aeimg-classify -a basic data/mnist-numbers
ls src/*/* | entr -rc aeimg-classify -a basic data/mnist-numbers-tiny
ls src/*/* | entr -rc aeimg-classify -a basic data/mnist-numbers-tiny --verbose
ls src/*/* | entr -rc aeimg-classify -a download_mnist_num data/test
aeimg-classify -a download_mnist_fashion data/fashion && aeimg-classify -a basic data/fashion

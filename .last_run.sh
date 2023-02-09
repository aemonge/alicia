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

# -- Rafactored

aeimg-classify test basic data/MNIST
aeimg-classify download MNIST data/MNIST
aeimg-classify download FashionMNIST data/MNIST && aeimg-classify --verbose test -n 4 basic data/MNIST
aeimg-classify download MNIST data/mnist/train data/mnist/test
rm -rf data/mnist && mkdir -p data/mnist/train && mkdir -p data/mnist/test
ls src/*/* | entr -rc aeimg-classify --verbose train basic data/mnist
rm -rf data/fashion/test data/fashion/train && mkdir -p data/fashion/test data/fashion/train;  aeimg-classify download FashionMNIST data/fashion/train data/fashion/test && aeimg-classify --verbose test -n 4 basic data/fashion
cat data/fashion/train/labels.csv| cut -f2 -d, | sort | uniq > fashion.classes.txt

# -- Rafactored

rm -rf data/mnist; mkdir -p data/mnist/train data/mnist/test; alicia -v download MNIST data/mnist/train data/mnist/test
alicia -v train basic data/mnist -s data/mnist.basic.pth -e 15
alicia classify -n 4 data/mnist.basic.pth out
rm -rf data/fashion ; mkdir data/fashion; alicia -v download FashionMNIST data/fashion
alicia create basic data/mnist-fashion/labels.csv data/basic.pth -u 7
alicia create basic data/mnist-fashion/labels.csv data/basic.pth && alicia info data/basic.pth
alicia train data/basic.pth data/mnist-fashion-small -b 4 -e 3 -l 0.003 -p
alicia create basic data/mnist-fashion-small/labels.csv data/basic.pth -i 784 && alicia info data/basic.pth
alicia create basic data/mnist-fashion-small/labels.csv data/basic.pth -i 784 && alicia info data/basic.pth && alicia train data/basic.pth data/mnist-fashion-small data/mnist-fashion-small/labels.csv -b 4 -e 7 -l 0.0003 -m 0.85 -p
alicia create basic data/mnist-fashion-small/labels.csv data/basic.pth -i 784 && alicia info data/basic.pth && alicia train data/basic.pth data/mnist-fashion-small data/mnist-fashion-small/labels.csv -b 4 -e 7 -l 0.0003 -m 0.85 -p
alicia create basic data/mnist-fashion-small/labels.csv data/basic.pth -i 784 && alicia info data/basic.pth && alicia train data/basic.pth data/mnist-fashion-small data/mnist-fashion-small/labels.csv -b 4 -e 37 -l 0.003 -m 0.85 && alicia test data/basic.pth data/mnist-fashion-small data/mnist-fashion-small/labels.csv -b 4 -h -n 9
rm -r data/mnist-fashion/[tv]* && alicia download FashionMNIST data/mnist-fashion
rm -r data/mnist-fashion/[tv]* && alicia download FashionMNIST data/mnist-fashion

#! /bin/bash

set -e
PTH="/tmp/model.pth"

# for MODEL in "Mnasnet" "Alexnet" "Squeezenet"
# do
#   alicia create $MODEL data/flowers/labels.csv $PTH -i $((228*228*3))
#   alicia info $PTH
#   alicia train $PTH data/flowers data/flowers/labels.csv -e 1 -t flowers_unshaped
#   alicia test $PTH data/flowers data/flowers/labels.csv -n 1 -t flowers_unshaped -c
# done
#
# for MODEL in "Basic" "Elemental"
# do
#   alicia create $MODEL data/mnist-fashion/labels.csv $PTH -i 784
#   alicia info $PTH
#   alicia train $PTH data/mnist-fashion data/mnist-fashion/labels.csv -e 1 -t mnist
#   alicia test $PTH data/mnist-fashion data/mnist-fashion/labels.csv -n 1 -t mnist -c
# done

alicia create Basic data/flowers/labels.csv $PTH -i $((228*228*3))
alicia info $PTH
alicia modify -i 28 $PTH -f 0 "Conv2d(1, 128, kernel_size=7, stride=2)"
alicia modify $PTH -f 1 "None"
alicia info $PTH

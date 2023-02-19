#!/bin/bash

MODEL="Basic"
DATA="MNIST"

echo "== ================================================================================================="
echo "== # Download                                                                                      ="
echo "== alicia download $DATA mnist-data"
echo "== ================================================================================================="
alicia download $DATA mnist-data

echo "== ================================================================================================="
echo "== # Create"
echo "== alicia create $MODEL mnist-data/labels.csv model-dropless.pth"
echo "== alicia create $MODEL mnist-data/labels.csv model.pth -d 0.8"
echo "== ================================================================================================="
alicia create $MODEL mnist-data/labels.csv model-dropless.pth
alicia create $MODEL mnist-data/labels.csv model.pth -d 0.8

echo "== ================================================================================================="
echo "== # Info"
echo "== alicia info model-dropless.pth"
echo "== alicia info model.pth"
echo "== ================================================================================================="
alicia info model.pth
alicia info model-dropless.pth

echo "== ================================================================================================="
echo "== # Compare Info"
echo "== alicia diff-info model*"
echo "== ================================================================================================="
alicia compare diff-info model*

echo "== ================================================================================================="
echo "== # Train"
echo "== alicia train model-dropless.pth mnist-data mnist-data/labels.csv -e 5"
echo "== alicia train model.pth mnist-data mnist-data/labels.csv -e 5"
echo "== ================================================================================================="
alicia train model-dropless.pth mnist-data mnist-data/labels.csv -e 5
alicia train model.pth mnist-data mnist-data/labels.csv -e 5

echo "== ================================================================================================="
echo "== # Compare accuracy"
echo "== alicia train model-dropless.pth mnist-data mnist-data/labels.csv -e 5"
echo "== alicia train model.pth mnist-data mnist-data/labels.csv -e 5"
echo "== ================================================================================================="
alicia compare accuracy -d mnist-data -c mnist-data/labels.csv model*

echo "== ================================================================================================="
echo "== # Test and visualize"
echo "== alicia test -n 4 -c model.pth mnist-data mnist-data/labels.csv"
echo "== alicia test -n 4 model-dropless.pth mnist-data mnist-data/labels.csv"
echo "== ================================================================================================="
alicia test -n 4 -c model.pth mnist-data mnist-data/labels.csv
alicia test -n 4 model-dropless.pth mnist-data mnist-data/labels.csv

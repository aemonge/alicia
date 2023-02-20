#!/bin/bash

rm -rf mnist-data 2> /dev/null
MODEL="Basic"
DATA="FashionMNIST"

slowprint() {
  IFS=''
  while read -n 1 c; do
      echo -e -n "$c"
    sleep 0.04
  done <<< "$1"
  echo ""
}

typecmd() {
  IFS=''
  echo -en '\e[31m$> \e[34m'
  while read -n 1 c; do
      echo -n "$c"
    sleep 0.04
  done <<< "$1"
  echo -e '\e[39m'
}

clear
echo "== ================================================================================================="
slowprint "== Welcome to Alicia, and this quick demo. Let's quickly review our \`--help\`"
echo "== ================================================================================================="
sleep 1
typecmd 'alicia --help'
alicia --help
sleep 3

clear
echo "== ================================================================================================="
slowprint "== You can use Alicia to download $DATA data."
echo "== ================================================================================================="
sleep 1
typecmd "alicia download $DATA mnist-data"
alicia download $DATA mnist-data
sleep 3

clear
echo "== ================================================================================================="
slowprint "== You can use Alicia to create a new model, to train. We will create a Basic model."
echo "== ================================================================================================="
sleep 1
typecmd "alicia create $MODEL mnist-data/labels.csv model.pth -i 784"
alicia create $MODEL mnist-data/labels.csv model.pth -i 784
sleep 3

clear
echo "== ================================================================================================="
slowprint "== Now let's print out the information about such model"
echo "== ================================================================================================="
sleep 1
typecmd "alicia info model.pth"
alicia info model.pth
sleep 3

clear
echo "== ================================================================================================="
slowprint "== Now, the fun part begins, training our newly created model."
slowprint "==      since training will take long time, I'll be using a smaller (pre download) data set to train"
slowprint "==      note that I'm choosing the \`data/mnist-fashion-small\` folder as main data folder"
echo "== ================================================================================================="
sleep 1
typecmd "alicia train model.pth data/mnist-fashion-small mnist-data/labels.csv -l 0.3 -e 6"
alicia train model.pth data/mnist-fashion-small mnist-data/labels.csv -l 0.03 -e 6
sleep 5

clear
echo "== ================================================================================================="
slowprint "== Now let's test it out, and check our accuracy with a geeky console image processing"
echo "== ================================================================================================="
sleep 1
typecmd "alicia test model.pth mnist-data mnist-data/labels.csv -n 3 -c "
alicia test model.pth data/mnist-fashion-small mnist-data/labels.csv -n 2 -c
sleep 8

clear
echo "== ================================================================================================="
slowprint "== Finally, don't forget to check \`--help\` on all commands to such as comparing models"
echo "== ================================================================================================="
sleep 1
typecmd "alicia compare --help"
alicia compare --help
sleep 3

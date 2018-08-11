#!/bin/bash

CLASSES=64
echo classes = $CLASSES

for (( c=0; c<$CLASSES; c++ ))
do python3 DNN_regression_train.py --index $c > output.txt; echo $c; done 
#!/bin/bash

# ./mnist_sbatch.sh mnist_init_train.txt

file=$1

while read line; do
    sbatch mnist_train.sh $line
done < $file

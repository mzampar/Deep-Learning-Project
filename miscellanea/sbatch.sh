#!/bin/bash

# ./sbatch.sh init_train.txt

file=$1

while read line; do
    sbatch train.sh $line
done < $file

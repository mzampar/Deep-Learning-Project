#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=GPU
#SBATCH --mem=1gb
#SBATCH --job-name=plot
#SBATCH -A dssc
#SBATCH --output=plot_%j.out

# Generate plots of the loss
src="/u/dssc/mzampar/Deep-Learning-Project/display"

cd "/u/dssc/mzampar/Deep-Learning-Project/mnist-models"

for folder in $(ls -d *); do
    cd $folder
    cat slurm_mnist_${folder}.out | grep -E "^Epoch \[[0-9]+/[0-9]+\], Batch \[[0-9]+\], Loss:|^Training with sequence length [0-9]+." | sed -E 's/^.*Loss: ([0-9]+\.[0-9]+)$/\1/' > slurm_mnist_${folder}_cleaned.out
    python $src/plot_loss_median.py --cleaned_file slurm_mnist_${folder}_cleaned.out --file slurm_mnist_${folder}.out --out_file loss-${folder}-median.png
    cd ..
done


cd "/u/dssc/mzampar/Deep-Learning-Project/rain-models"

for folder in $(ls -d *); do
    cd $folder
    cat slurm_rain_${folder}.out | grep -E "^Epoch \[[0-9]+/[0-9]+\], Batch \[[0-9]+\], Loss:|^Training with sequence length [0-9]+." | sed -E 's/^.*Loss: ([0-9]+\.[0-9]+)$/\1/' > slurm_rain_${folder}_cleaned.out
    python $src/plot_loss_median.py --cleaned_file slurm_rain_${folder}_cleaned.out --file slurm_rain_${folder}.out --out_file loss-${folder}-median.png
    cd ..
done
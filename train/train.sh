#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --gpus=1
#SBATCH --mem=100gb
#SBATCH --job-name=train_conv_lstm
#SBATCH -A dssc



srun python -u train.py --job_id $SLURM_JOB_ID --num_hidden "16,8,8,16" --stride 2 --filter_size 3 --batch_size 64 --num_epochs 1 --layer_norm True --schedule_sampling True --schedule False --ssim False
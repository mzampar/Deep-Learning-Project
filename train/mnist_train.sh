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


srun python -u mnist_train.py --job_id $SLURM_JOB_ID --num_hidden 64 32 32 16 --stride 2 --filter_size 5 3 3 3 --batch_size 64 --max_pool 1 --leaky_slope 0.2 --bias 0 --transpose 1 --num_epochs 1 --layer_norm 1 --schedule_sampling 1 --schedule 0 --loss 1 --initial_lr 0.01 --gamma 0.5
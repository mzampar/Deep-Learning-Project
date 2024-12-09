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


srun python -u train.py --job_id $SLURM_JOB_ID --num_hidden "32,16,16,32" --stride 2 --filter_size 5 --batch_size 32 --bias 1 --transpose 1 --num_epochs 1 --layer_norm 1 --schedule_sampling 1 --schedule 0 --loss 1 --initial_lr 1.0 --gamma 0.5
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
#SBATCH --output=slurm_mnist_%j.out

# Boolean options: --max_pooling, --bias, --transpose, --layer_norm, --schedule_sampling and --schedule

srun python -u mnist_train.py --job_id $SLURM_JOB_ID --num_hidden 64 32 32 16 --stride 2 --filter_size 5 3 3 3 --batch_size 64 --max_pool --leaky_slope 0.2 --transpose --num_epochs 1 --layer_norm --schedule_sampling --loss 1 --initial_lr 0.01 --gamma 0.5
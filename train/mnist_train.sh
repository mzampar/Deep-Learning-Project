#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --gpus=1
#SBATCH --mem=100gb
#SBATCH --job-name=train_conv_lstm
#SBATCH -A dssc
#SBATCH --output=slurm_mnist_%j.out

echo "Starting job $SLURM_JOB_ID"

# Boolean options: --max_pooling, --bias, --transpose, --layer_norm, --schedule_sampling, --schedule, --use_lstm_output

out_folder="/u/dssc/mzampar/Deep-Learning-Project/mnist-models/$SLURM_JOB_ID"
mkdir -p $out_folder

num_hidden="64 32 32 16"
stride="2"
filter_size="5 3 3 3"
batch_size="64"
leaky_slope="0.2"
transpose="--transpose"
num_epochs="1"
layer_norm="--layer_norm"
schedule_sampling="--schedule_sampling"
loss="1"
initial_lr="0.01"
gamma="0.5"
model_name="$out_folder/model_$SLURM_JOB_ID.pth"

srun python -u mnist_train.py --job_id $SLURM_JOB_ID --num_hidden $num_hidden --stride $stride --filter_size $filter_size --batch_size $batch_size --max_pool --leaky_slope $leaky_slope $transpose --num_epochs $num_epochs $layer_norm $schedule_sampling --loss $loss --initial_lr $initial_lr --gamma $gamma --model_name $model_name

mv slurm_mnist_$SLURM_JOB_ID.out $out_folder

# Generate plots of the loss
src="/u/dssc/mzampar/Deep-Learning-Project/display"
python $src/plot_loss.py --file $out_folder/slurm_mnist_$SLURM_JOB_ID.out --out_file $out_folder/loss-$SLURM_JOB_ID.png 

python $src/mnist_generate_gif.py --model $model_name --out_folder $out_folder --num_hidden $num_hidden --stride $stride --filter_size $filter_size --max_pool --leaky_slope $leaky_slope $transpose $layer_norm --job_id $SLURM_JOB_ID --fig_height 64

rm *.gif
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --mem=100gb
#SBATCH --job-name=train_conv_lstm
#SBATCH -A dssc
#SBATCH --output=slurm_rain_%j.out

echo "Starting job $SLURM_JOB_ID"

# Boolean options: --max_pooling, --bias, --transpose, --layer_norm, --schedule_sampling, --schedule, --use_lstm_output

out_folder="/u/dssc/mzampar/Deep-Learning-Project/rain-models/$SLURM_JOB_ID"
mkdir -p $out_folder

num_hidden="64 32 32 16"
stride="2"
filter_size="3 3 3 3"
batch_size="64"
leaky_slope="0.2"
transpose="--transpose"
num_epochs="1"
layer_norm="--layer_norm"
schedule_sampling="--schedule_sampling"
loss="1"
initial_lr="0.01"
gamma="0.5"
max_pool="--max_pool"
bias="--bias"
schedule="--schedule"
use_lstm_output="--use_lstm_output"

model_name="model_$SLURM_JOB_ID.pth"

#srun python -u train.py --job_id $SLURM_JOB_ID --num_hidden $num_hidden --stride $stride --filter_size $filter_size --batch_size $batch_size --leaky_slope $leaky_slope --num_epochs $num_epochs --loss $loss --initial_lr $initial_lr --gamma $gamma --model_name $out_folder/$model_name $transpose $layer_norm $max_pool $bias

# Generate plots of the loss
src="/u/dssc/mzampar/Deep-Learning-Project/display"

cat slurm_rain_$SLURM_JOB_ID.out | grep -E "^Epoch \[[0-9]+/[0-9]+\], Batch \[[0-9]+\], Loss:|^Training with sequence length [0-9]+." | sed -E 's/^.*Loss: ([0-9]+\.[0-9]+)$/\1/' > slurm_rain_${SLURM_JOB_ID}_cleaned.out

#python $src/plot_loss.py --file slurm_rain_$SLURM_JOB_ID.out --out_file $out_folder/loss-$SLURM_JOB_ID.png
python $src/plot_loss_median.py --cleaned_file slurm_rain_${SLURM_JOB_ID}_cleaned.out --file slurm_rain_$SLURM_JOB_ID.out --out_file $out_folder/loss-$SLURM_JOB_ID-median.png

python $src/generate_gif.py --fig_height 128 --model $model_name --out_folder $out_folder --job_id $SLURM_JOB_ID --num_hidden $num_hidden --stride $stride --filter_size $filter_size --leaky_slope $leaky_slope $transpose $max_pool $bias $layer_norm

rm *.gif

mv slurm_rain_$SLURM_JOB_ID.out slurm_rain_${SLURM_JOB_ID}_cleaned.out $out_folder
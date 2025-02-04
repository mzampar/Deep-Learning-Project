import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Add the relative path to the system path
sys.path.append(os.path.abspath("u/dssc/mzampar/Deep_Learning_Project/train"))

from PIL import Image, ImageDraw, ImageFont, ImageSequence

import torch as th
import torch.nn as nn
from torchvision import transforms

from ConvLSTM_model import ConvLSTM_Model


# Define function to add a border and title to a PIL image
def add_border_and_title(image, title, border_size=10, font_size=20, color="white"):
    # Add border
    width, height = image.size
    new_width = width + 2 * border_size
    new_height = height + 2 * border_size + font_size + 5  # Additional space for the title
    bordered_image = Image.new("RGB", (new_width, new_height), color=color)
    bordered_image.paste(image, (border_size, border_size + font_size + 5))
    # Add title text
    draw = ImageDraw.Draw(bordered_image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(((new_width-font_size)//4,border_size), title, fill="black", font=font)
    return bordered_image

# function to generate gif
def generate_gif(input_frames, output_frames, model, filename, predict):

    input_pil_frames = [transforms.ToPILImage()(frame) for frame in input_frames]
    true_pil_frames = [transforms.ToPILImage()(frame) for frame in output_frames]

    true_gif_list = input_pil_frames + true_pil_frames
    #true_gif_list[0].save(filename, save_all=True, append_images=true_gif_list[1:], duration=10, loop=0)

    input_pil_frames = [
        add_border_and_title(transforms.ToPILImage()(frame), "Input Frame")
        for frame in input_frames
    ]
    true_pil_frames = [
        add_border_and_title(transforms.ToPILImage()(frame), "Target Frame")
        for frame in output_frames
    ]

    if not predict:
        # List of frames as PIL Image objects
        # Duration sets the display time for each frame in milliseconds
        # Loop sets the number of loops. Default is 0 and means infinite
        true_gif_list = input_pil_frames + true_pil_frames
        true_gif_list[0].save(
            filename, save_all=True, append_images=true_gif_list[1:], duration=1000, loop=0
        )
    else:
        input_frames = th.stack(input_frames, dim=0)
        input_frames = input_frames.unsqueeze(0)
        mask_true = th.ones_like(input_frames)
        predicted_frames = model(input_frames, mask_true, schedule_sampling=False)
        input_frames = input_frames.squeeze(0)
        predicted_frames = predicted_frames.squeeze(0)
        print(predicted_frames.shape)
        predicted_pil_frames = [transforms.ToPILImage()(frame) for frame in predicted_frames]

        pred_gif_list = input_pil_frames + predicted_pil_frames

        predicted_pil_frames = [
            add_border_and_title(transforms.ToPILImage()(frame), "Predicted Frame", color="red")
            for frame in predicted_frames
        ]

        pred_gif_list = input_pil_frames + predicted_pil_frames
        pred_gif_list[0].save(
            filename, save_all=True, append_images=pred_gif_list[1:], duration=1000, loop=0
        )

# function to concatenate gifs
def concatenate_gifs_horizontally(gif1_path, gif2_path, output_path):
    # Open the two GIFs
    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)
    
    # Ensure the two GIFs have the same frame count
    frames1 = [frame.copy() for frame in ImageSequence.Iterator(gif1)]
    frames2 = [frame.copy() for frame in ImageSequence.Iterator(gif2)]
    max_frames = max(len(frames1), len(frames2))
    
    # Repeat frames if necessary to match lengths
    frames1 *= (max_frames // len(frames1)) + 1
    frames2 *= (max_frames // len(frames2)) + 1
    
    # Resize to ensure consistent heights
    common_height = max(gif1.height, gif2.height)
    frames1 = [frame.resize((frame.width, common_height)) for frame in frames1[:max_frames]]
    frames2 = [frame.resize((frame.width, common_height)) for frame in frames2[:max_frames]]
    
    # Concatenate frames horizontally
    concatenated_frames = []
    for frame1, frame2 in zip(frames1, frames2):
        new_width = frame1.width + frame2.width
        new_frame = Image.new("RGBA", (new_width, common_height))
        new_frame.paste(frame1, (0, 0))
        new_frame.paste(frame2, (frame1.width, 0))
        concatenated_frames.append(new_frame)
    
    # Save the concatenated GIF
    concatenated_frames[0].save(
        output_path,
        save_all=True,
        append_images=concatenated_frames[1:],
        loop=0,
        duration=gif1.info.get("duration", 100)
    )

parser = argparse.ArgumentParser()

# Action store_true means that if the argument is present, the value is True, otherwise is False
parser.add_argument('--job_id', type=str, required=True, help='SLURM job ID')
parser.add_argument('--num_hidden', type=int, nargs='+', required=True, help='List of hidden layer sizes')
parser.add_argument('--filter_size', type=int, nargs='+', required=True, help='List of filter sizes')
parser.add_argument('--stride', type=int, required=True, help='Stride')
parser.add_argument('--patch_size', type=int, required=False, default=1, help='Patch size')
parser.add_argument('--bias', action='store_true', help='Use bias')
parser.add_argument('--leaky_slope', type=float, required=True, help='Leaky ReLU slope')
parser.add_argument('--max_pooling', action='store_true', help='Enable max pooling')
parser.add_argument('--transpose', action='store_true', help='Enable transposition')
parser.add_argument('--use_lstm_output', action='store_true', help='Use LSTM output')
parser.add_argument('--layer_norm', action='store_true', help='Enable layer normalization')

parser.add_argument('--fig_height', type=int, required=True, help='Output folder')
parser.add_argument('--model', type=str, required=True, help='Output folder')
parser.add_argument('--out_folder', type=str, required=True, help='Output folder')


args = parser.parse_args()

# Assign values directly from args
job_id = args.job_id
leaky_slope = args.leaky_slope
max_pool = args.max_pooling
bias = args.bias
transpose = args.transpose
use_lstm_output = args.use_lstm_output
stride = args.stride
filter_size = args.filter_size
patch_size = args.patch_size
num_hidden = args.num_hidden
num_layers = len(num_hidden) if num_hidden else 0
layer_norm = args.layer_norm

fig_height = args.fig_height
model_name = args.model
out_folder = args.out_folder

id_data = pd.read_csv('/u/dssc/mzampar/Deep_Learning_Project/data/id_df_final_10.csv')

seq_len = id_data.groupby('sequence').size()
seq_len = seq_len.to_dict()
seq_rain = id_data.groupby('sequence')['rain_category'].mean()
seq_rain = seq_rain.to_dict()

seq_df = pd.DataFrame({'seq_len': seq_len, 'seq_rain': seq_rain})

# split the sequences in train and test set (80/20)
train_seq = seq_df.sample(frac=0.8, random_state=1)
test_seq = seq_df.drop(train_seq.index)

# get the sequences of the train and test set
train_seq_idx = train_seq.index
test_seq_idx = test_seq.index
train_data = id_data[id_data['sequence'].isin(train_seq_idx)]
test_data = id_data[id_data['sequence'].isin(test_seq_idx)]

ids = [test_data.iloc[i]["id"] for i in range(20)]
test_images = [f"tensor_{id}.pt" for id in ids]

ids = [train_data.iloc[i]["id"] for i in range(10,30)]
train_images = [f"tensor_{id}.pt" for id in ids]

batch_size = 1
num_layers = len(num_hidden)
custom_model_config = {
    'in_shape': [1, fig_height, fig_height], # C, H, W
    'stride': stride, 
    'filter_size': filter_size, 
    'leaky_slope' : leaky_slope,
    'max_pool' : max_pool,
    'layer_norm' : layer_norm,
    'bias' : bias,
    'transpose' : transpose,
    'patch_size': 1
}

if th.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Instantiate the model
input_dim = 3  # Assuming x_train shape is (batch_size, sequence_length, channels, height, width)
model = ConvLSTM_Model(num_layers, num_hidden, custom_model_config)

# Load the model
# Load the state dictionary
state_dict = th.load(model_name, map_location=th.device('cpu'), weights_only=True)

# Remove `module.` prefix if it exists
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# Load the adjusted state dict into the model
model.load_state_dict(new_state_dict, strict=True)
folder="/u/dssc/mzampar/scratch/grey_tensor"
input_frames = [th.load(f"{folder}/{image}", weights_only=True) for image in train_images[0:10]]
output_frames = [th.load(f"{folder}/{image}", weights_only=True) for image in train_images[10:]]
generate_gif(input_frames=input_frames, output_frames=output_frames, model=model, filename="train_pred.gif", predict=True)
generate_gif(input_frames=input_frames, output_frames=output_frames, model=model, filename="train_true.gif", predict=False)
concatenate_gifs_horizontally("train_true.gif", "train_pred.gif", out_folder + f"/train_{model_name}.gif")

input_frames = [th.load(f"{folder}/{image}", weights_only=True) for image in test_images[0:10]]
output_frames = [th.load(f"{folder}/{image}", weights_only=True) for image in test_images[10:]]
generate_gif(input_frames=input_frames, output_frames=output_frames, model=model, filename="test_pred.gif", predict=True)
generate_gif(input_frames=input_frames, output_frames=output_frames, model=model, filename="test_true.gif", predict=False)
concatenate_gifs_horizontally("test_true.gif", "test_pred.gif", out_folder + f"/test_{model_name}.gif")
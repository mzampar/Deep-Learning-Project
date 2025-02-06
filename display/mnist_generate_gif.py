import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Add the relative path to the system path
sys.path.append("/u/dssc/mzampar/Deep-Learning-Project/train")

from PIL import Image, ImageDraw, ImageFont, ImageSequence

import torch as th
import torch.nn as nn
from torchvision import transforms

from ConvLSTM_model import ConvLSTM_Model
from utils import MnistSequenceDataset


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
def generate_gif(index, dataset, model, filename, predict):

    input_frames = dataset[index][0]
    output_frames = dataset[index][1]

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
        input_frames = input_frames.unsqueeze(0)
        mask_true = th.ones_like(input_frames)
        predicted_frames = model(input_frames, mask_true, schedule_sampling=False)
        input_frames = input_frames.squeeze(0)
        predicted_frames = predicted_frames.squeeze(0)
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


seq_len = 9
data = np.load("/u/dssc/mzampar/scratch/mnist_test_seq.npy").astype(np.float32)/255
train_idx = int(data.shape[1] * 0.8)
train_dataset = MnistSequenceDataset(data[:,:train_idx], seq_len, seq_len)
test_dataset = MnistSequenceDataset(data[:,train_idx:], seq_len, seq_len)

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
    'patch_size': 1,
    'use_lstm_output': use_lstm_output
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
state_dict = th.load(out_folder + f"/{model_name}", map_location=th.device('cpu'), weights_only=True)
# Remove `module.` prefix if it exists
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# Load the adjusted state dict into the model
model.load_state_dict(new_state_dict, strict=True)

generate_gif(0, train_dataset, model, filename="train_pred.gif", predict=True)
generate_gif(0, train_dataset, model, filename="train_true.gif", predict=False)
concatenate_gifs_horizontally("train_true.gif", "train_pred.gif", out_folder + f"/mnist_train_{model_name}.gif")


generate_gif(1000, test_dataset, model, filename="test_pred.gif", predict=True)
generate_gif(1000, test_dataset, model, filename="test_true.gif", predict=False)
concatenate_gifs_horizontally("test_true.gif", "test_pred.gif", out_folder + f"/mnist_test_{model_name}.gif")

import torch as th
from torch.utils.data import Dataset
from torch import nn
from ConvLSTM_model import ConvLSTM_Model
from utils import SSIM_MSE_Loss, MnistSequenceDataset
import pandas as pd
import numpy as np
import argparse
import time

# Default values
stride = 2
filter_size = 3
patch_size = 1
num_hidden = [16,8,8,16]
num_layers = len(num_hidden)
batch_size = 64
layer_norm = True
schedule_yes = False
schedule_sampling = False
num_epochs = 1
criterion = nn.MSELoss()
initial_lr = 0.1
gamma = 0.5
bias = False
transpose = True
leaky_slope = None
max_pool = False
loss = 0
use_lstm_output = False

parser = argparse.ArgumentParser()

# Action store_true means that if the argument is present, the value is True, otherwise is False
parser.add_argument('--job_id', type=str, required=True, help='SLURM job ID')
parser.add_argument('--schedule', action='store_true', help='Enable scheduler')
parser.add_argument('--num_hidden', type=int, nargs='+', required=True, help='List of hidden layer sizes')
parser.add_argument('--filter_size', type=int, nargs='+', required=True, help='List of filter sizes')
parser.add_argument('--stride', type=int, required=True, help='Stride')
parser.add_argument('--patch_size', type=int, required=False, default=1, help='Patch size')
parser.add_argument('--bias', action='store_true', help='Use bias')
parser.add_argument('--leaky_slope', type=float, required=True, help='Leaky ReLU slope')
parser.add_argument('--max_pooling', action='store_true', help='Enable max pooling')
parser.add_argument('--transpose', action='store_true', help='Enable transposition')
parser.add_argument('--use_lstm_output', action='store_true', help='Use LSTM output')
parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs')
parser.add_argument('--loss', type=int, choices=[0, 1, 2], required=True, help='Loss: 0 = MSE, 1 = BCE, 2 = SSIM+MSE')
parser.add_argument('--layer_norm', action='store_true', help='Enable layer normalization')
parser.add_argument('--schedule_sampling', action='store_true', help='Enable schedule sampling')
parser.add_argument('--initial_lr', type=float, required=False, default=0.01, help='Initial learning rate')
parser.add_argument('--gamma', type=float, required=False, help='Gamma for scheduler')
parser.add_argument('--model_name', type=str, required=False, help='Model file')

args = parser.parse_args()

# Assign values directly from args
job_id = args.job_id
schedule_yes = args.schedule
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
batch_size = args.batch_size
num_epochs = args.num_epochs
layer_norm = args.layer_norm
schedule_sampling = args.schedule_sampling
initial_lr = args.initial_lr
gamma = args.gamma
model_name = args.model_name

# Keep the conditional for loss
if args.loss is not None:
    loss = args.loss
    if args.loss == 0:
        criterion = nn.MSELoss()
    elif args.loss == 1:
        criterion = nn.BCELoss(reduction='sum')
    elif args.loss == 2:
        criterion = SSIM_MSE_Loss()

print(f"Training with:\n    architecture = {num_hidden},\n    stride = {stride},\n    filter_size = {filter_size},\n    leaky_slope = {leaky_slope},\n    max_pool = {max_pool},\n    layer norm = {layer_norm},\n    loss = {criterion},\n    batch size = {batch_size},\n    num_epochs = {num_epochs},\n    scheduled_sampling = {schedule_sampling},\n    bias = {bias},\n    transpose = {transpose},\n    use_lstm_output = {use_lstm_output},\n    scheduler = {schedule_yes},\n    initial_lr = {initial_lr},\n    gamma = {gamma}.")
print("")

custom_model_config = {
    'in_shape': [1, 64, 64], # C, H, W
    'patch_size': patch_size,
    'filter_size': filter_size, 
    'stride': stride, 
    'layer_norm' : layer_norm, 
    'transpose': transpose,
    'bias': bias, 
    'leaky_slope': leaky_slope,
    'max_pool': max_pool,
    'use_lstm_output': use_lstm_output
}

if th.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")
device = th.device("cuda" if th.cuda.is_available() else "cpu")
th.cuda.empty_cache()

# Instantiate the model
# x_train shape is (batch_size, sequence_length, channels, height, width)
model = ConvLSTM_Model(num_layers, num_hidden, custom_model_config)
# model = nn.DataParallel(model)
model.to(device)
# Define loss and optimizer
alpha = 1.0
# Add a learning rate scheduler
if schedule_yes:
    optimizer = th.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    print(f"Using learning rate scheduler with initial_lr = {initial_lr} and gamma = {gamma}.")
else:
    optimizer = th.optim.Adam(model.parameters(), lr=initial_lr)

if schedule_sampling:
    mask_true = th.ones(custom_model_config['in_shape'])
    mask_true = mask_true.to(device)
else:
    mask_true = None

data = np.load('/u/dssc/mzampar/scratch/mnist_test_seq.npy').astype(np.float32)/255
print(f"Data shape: {data.shape}")
train_percentage = 0.9
train_idx = int(data.shape[1] * train_percentage)

start = time.time()
# Loop over the dataset multiple times, with different sequence lengths to avoid the vanishing gradient problem
max_seq_len = 9
for seq_len in range(2, max_seq_len+1):
    print("")
    th.cuda.empty_cache()
    if loss==2:
        alpha -= 0.05
        criterion = SSIM_MSE_Loss(alpha=alpha)
        print(f"Training with sequence length {seq_len}, with alpha = {alpha}.")
    else:
        print(f"Training with sequence length {seq_len}.")

    # After each sequence length, we reset the optimizer and the scheduler
    if schedule_yes:
        optimizer = th.optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    # To extract the validation set, we tried to select some random indices,
    # But in this way numpy creates a copy and we run out of memory, 
    # So we work with 2 indexes for the validation set
    """
    # Compute index ranges
    num_samples = int(0.1 * data.shape[1])  # 10% for valiation
    # Generate random validation indices from the first 90% of data
    np.random.seed(seq_len) 
    random_indexes = np.random.choice(range(train_idx), size=num_samples, replace=False)
    # Get training indexes: all indexes < max_index but not in random_indexes
    train_indexes = np.setdiff1d(np.arange(train_idx), random_indexes)
    """ 

    num_samples = int(0.1 * data.shape[1])  # 10% for valiation
    validation_index_end = np.random.randint(num_samples, train_idx)
    validation_index_start = validation_index_end - num_samples

    # Create train, validation and test datasets
    # Training data (everything before validation starts + everything after validation ends)
    train_dataset = MnistSequenceDataset(
        np.concatenate([data[:,:validation_index_start,:,:], data[:,validation_index_end:,:,:]], axis=1), seq_len, seq_len
    )
    # Validation data (data between validation start and end indices)
    validation_dataset = MnistSequenceDataset(data[:,validation_index_start:validation_index_end], seq_len, seq_len)
    test_dataset = MnistSequenceDataset(data[:,train_idx:], seq_len, seq_len)
    dataloader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    validation_dataloader = th.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Number of elements to set to zero in the mask
    total_pixels = custom_model_config['in_shape'][1] * custom_model_config['in_shape'][2]
    num_zeros = int(total_pixels * ( seq_len / (max_seq_len * 4) ))

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if schedule_sampling:
                flat_mask = mask_true.view(-1)
                # Randomly choose indices to set to zero
                zero_indices = th.randperm(flat_mask.numel())[:num_zeros]
                flat_mask[zero_indices] = 0
                mask_true = flat_mask.view(custom_model_config['in_shape'])
            inputs, targets = inputs.to(device), targets.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs, mask_true = mask_true, schedule_sampling=schedule_sampling)
            # Compute loss
            loss = criterion(outputs, targets)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.6f}")

            del outputs, loss, inputs, targets
            th.cuda.empty_cache()

        if schedule_yes:
            scheduler.step()

        epoch_train_loss = running_loss / len(dataloader)
        print(f"Seq_Len: {seq_len}, Epoch [{epoch+1}/{num_epochs}] - Average Train Loss: {epoch_train_loss:.4f}")

        # Test phase
        model.eval()
        test_loss = 0.0
        with th.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, th.ones_like(inputs), schedule_sampling=False)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        epoch_test_loss = test_loss / len(test_dataloader)
        print(f"Seq_Len: {seq_len}, Epoch [{epoch+1}/{num_epochs}] - Average Test Loss: {epoch_test_loss:.4f}")
        print("Elapsed time: {:.2f} seconds".format(time.time() - start))

        # Validation phase
        model.eval() 
        val_loss = 0.0

        with th.no_grad():
            for batch_idx, (inputs, targets) in enumerate(validation_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, th.ones_like(inputs), schedule_sampling=False)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(validation_dataloader)
        print(f"Seq_Len: {seq_len}, Epoch [{epoch+1}/{num_epochs}] - Average Validation Loss: {epoch_val_loss:.4f}")
        print("Elapsed time: {:.2f} seconds".format(time.time() - start))


print("")
print("Training complete!")
print("Totoal elapsed time: {:.2f} seconds".format(time.time() - start))

th.save(model.state_dict(), model_name)
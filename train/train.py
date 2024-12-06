import torch as th
from torch.utils.data import Dataset
from torch import nn

from ConvLSTM_model import ConvLSTM_Model
from utils import SequenceDataset, SSIM_MSE_Loss

import pandas as pd
import os

# define a dataset
id_data = pd.read_csv('../data/id_df_final.csv')

seq_len = id_data.groupby('sequence').size()
seq_len = seq_len.to_dict()
seq_rain = id_data.groupby('sequence')['rain_category'].mean()
seq_rain = seq_rain.to_dict()

seq_df = pd.DataFrame({'seq_len': seq_len, 'seq_rain': seq_rain})

# split the sequences in train and test set (80/20)
train_seq = seq_df.sample(frac=0.8, random_state=1)
test_seq = seq_df.drop(train_seq.index)

print(train_seq['seq_len'].mean(), test_seq['seq_len'].mean())
print(train_seq['seq_len'].std(), test_seq['seq_len'].std())
print(train_seq['seq_rain'].mean(), test_seq['seq_rain'].mean())
print(train_seq['seq_rain'].std(), test_seq['seq_rain'].std())

# get the sequences of the train and test set
train_seq_idx = train_seq.index
test_seq_idx = test_seq.index

train_data = id_data[id_data['sequence'].isin(train_seq_idx)]
train_data.shape

test_data = id_data[id_data['sequence'].isin(test_seq_idx)]
test_data.shape

id_data = None
seq_len = None
seq_rain = None
seq_df = None
train_seq = None
test_seq = None
train_seq_idx = None
test_seq_idx = None

# model
filter_size = 5
stride = 1
patch_size = 2
layer_norm = 0

num_hidden = [32, 64, 64, 32]
num_layers = len(num_hidden)

custom_model_config = {
    'in_shape': [1, 128, 128], # T, C, H, W
    'patch_size': 1,
    'filter_size': 1, # given to ConvLSTMCell
    'stride': 1, # given to ConvLSTMCell
    'layer_norm' : False, # given to ConvLSTMCell
    # the sum of pre_seq_length and aft_seq_length has to be = len(inputs)
    'reverse_scheduled_sampling': 0
}

if th.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

device = th.device("cuda" if th.cuda.is_available() else "cpu")
th.cuda.empty_cache()

# Instantiate the model
# Assuming x_train shape is (batch_size, sequence_length, channels, height, width)
model = ConvLSTM_Model(num_layers, num_hidden, custom_model_config, schedule_sampling=False)
#model = nn.DataParallel(model)
model.to(device)
# Define loss and optimizer
criterion = nn.MSELoss()
criterion = SSIM_MSE_Loss(alpha=0.5)
optimizer = th.optim.Adam(model.parameters())

# Loop over the dataset multiple times, with different sequence lengths to avoid the vanishing gradient problem
train_losses_out = []
test_losses_out = []

for seq_len in range(2, 11):
    th.cuda.empty_cache()
    print(f"Training with sequence length {seq_len}")
    
    train_dataset = SequenceDataset(train_data, '../../scratch/grey_tensor/', seq_len, seq_len)
    test_dataset = SequenceDataset(test_data, '../../scratch/grey_tensor/', seq_len, seq_len)
    dataloader = th.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = th.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Training loop
    num_epochs = 1  # Set the number of epochs
    # Lists to keep track of the losses for each epoch
    train_losses = []
    test_losses = []

    mask_true = th.ones(custom_model_config['in_shape'])
    mask_true = mask_true.to(device)
    # Number of elements to set to zero
    num_zeros = seq_len * 1000

    for epoch in range(num_epochs*seq_len//2):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            """
            # Flatten the tensor to 1D
            flat_mask = mask_true.view(-1)
            # Randomly choose indices to set to zero
            zero_indices = th.randperm(flat_mask.numel())[:num_zeros]
            # Set those indices to zero
            flat_mask[zero_indices] = 0
            # Reshape back to the original shape
            mask_true = flat_mask.view(custom_model_config['in_shape'])
            """
            # Move data to device (GPU if available)
            inputs, targets = inputs.to(device), targets.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs, mask_true = mask_true, schedule_sampling=False)
            # Compute loss
            loss = criterion(outputs, targets)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # Accumulate loss
            running_loss += loss.item()
            
            # Print training info every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.4f}")

            del outputs, loss, inputs, targets
            th.cuda.empty_cache()
        # Calculate and store the average training loss for this epoch
        epoch_train_loss = running_loss / len(dataloader)
        train_losses.append(epoch_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Train Loss: {epoch_train_loss:.4f}")

        # Validation (test) phase
        model.eval()
        test_loss = 0.0
        with th.no_grad():  # No gradients needed for testing
            for batch_idx, (inputs, targets) in enumerate(test_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, th.ones_like(inputs), schedule_sampling=False)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        # Calculate and store the average test loss for this epoch
        epoch_test_loss = test_loss / len(test_dataloader)  # Using len(test_dataloader) for batch average
        test_losses.append(epoch_test_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Test Loss: {epoch_test_loss:.4f}")
    
    # Store the losses for this sequence length
    train_losses_out.append(train_losses)
    test_losses_out.append(test_losses)

print("Training complete!")


th.save(model.state_dict(), "../models/model.pth")

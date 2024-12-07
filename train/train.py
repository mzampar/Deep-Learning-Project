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

num_hidden = [16,8,8,16]
num_layers = len(num_hidden)
batch_size = 64
schedule_sampling = False

custom_model_config = {
    'in_shape': [1, 128, 128], # T, C, H, W
    'patch_size': 1,
    'filter_size': 3, # given to ConvLSTMCell
    'stride': 2, # given to ConvLSTMCell
    'layer_norm' : False, # given to ConvLSTMCell
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
model = ConvLSTM_Model(num_layers, num_hidden, custom_model_config)
#model = nn.DataParallel(model)
model.to(device)
# Define loss and optimizer
criterion = nn.MSELoss()
alpha = 1.0
criterion = SSIM_MSE_Loss(alpha=1.0)
print("Loss function: SSIM_MSE_Loss.")

# Add a learning rate scheduler
schedule_yes = False
if schedule_yes:
    initial_lr = 0.1
    optimizer = th.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
else:
    optimizer = th.optim.Adam(model.parameters())

print(f"Training with {num_hidden} architecture, batch size = {batch_size}, with scheduled_sampling = {schedule_sampling}, with scheduler = {schedule_yes}.")

if schedule_sampling:
    mask_true = th.ones(custom_model_config['in_shape'])
    mask_true = mask_true.to(device)
else:
    mask_true = None

# Loop over the dataset multiple times, with different sequence lengths to avoid the vanishing gradient problem
for seq_len in range(2, 21):
    alpha -= 0.05
    criterion = SSIM_MSE_Loss(alpha=alpha)
    th.cuda.empty_cache()
    print(f"Training with sequence length {seq_len}, with alpha = {alpha}.")

    train_dataset = SequenceDataset(train_data, '../../scratch/grey_tensor/', seq_len, seq_len)
    test_dataset = SequenceDataset(test_data, '../../scratch/grey_tensor/', seq_len, seq_len)
    dataloader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 1
    # Number of elements to set to zero in the mask
    num_zeros = seq_len * 100

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
            # Accumulate loss
            running_loss += loss.item()

            # Print training info every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.4f}")

            del outputs, loss, inputs, targets
            th.cuda.empty_cache()
        # scheduler.step()
        # Calculate and store the average training loss for this epoch
        epoch_train_loss = running_loss / len(dataloader)
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
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Test Loss: {epoch_test_loss:.4f}")

print("Training complete!")


th.save(model.state_dict(), "../models/model_16_8_8_16.pth")
import torch as th
from torch.utils.data import Dataset
from torch import nn
from ConvLSTM_model import ConvLSTM_Model
from utils import SequenceDataset, SSIM_MSE_Loss
import pandas as pd
import argparse
import time

# Default values
stride = 2
filter_size = 3
patch_size = 1
num_hidden = [16,8,8,16] # Number of hidden units in each layer
num_layers = len(num_hidden) # Number of conv cells
batch_size = 64
layer_norm = True
schedule_yes = False
schedule_sampling = False
num_epochs = 1
criterion = nn.MSELoss()
initial_lr = 0.1
gamma = 0.5
bias = True
transpose = True
leaky_slope = None
max_pool = False 


parser = argparse.ArgumentParser()

# Action store_true means that if the argument is present, the value is True, otherwise it is False

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
parser.add_argument('--out_folder', type=str, required=True, help='Output folder')


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
out_folder = args.out_folder

# Keep the conditional for loss
if args.loss is not None:
    loss = args.loss
    if args.loss == 0:
        criterion = nn.MSELoss()
    elif args.loss == 1:
        criterion = nn.BCELoss(reduction='sum')
    elif args.loss == 2:
        criterion = SSIM_MSE_Loss()


print(f"Training with:\n    architecture = {num_hidden},\n    stride = {stride},\n    filter_size = {filter_size},\n    leaky_slope = {leaky_slope},\n    max_pool = {max_pool},\n    layer norm = {layer_norm},\n    loss = {criterion},\n    batch size = {batch_size},\n    num_epochs = {num_epochs},\n    scheduled_sampling = {schedule_sampling},\n    scheduler = {schedule_yes},\n    bias = {bias},\n    transpose = {transpose},\n    use_lstm_output = {use_lstm_output},\n    initial_lr = {initial_lr},\n    gamma = {gamma}.")
print("")

# Define the model configuration
custom_model_config = {
    'in_shape': [1, 128, 128], # channels, height, width of the input tensor 
    'patch_size': patch_size, # patch size, to process different parts of the image
    'filter_size': filter_size, # A list, filter size of each convolution, same lenght as num_hidden
    'stride': stride, # stride of the convolution
    'layer_norm' : layer_norm, # wheter to use LayerNorm or not
    'transpose': transpose, # wheter to use transposed convolution or not
    'bias': bias, # wheter to use bias or not
    'leaky_slope': leaky_slope, # slope of the LeakyReLU, if None, LeakyReLU is not used
    'max_pool': max_pool, # wheter to use max pooling or not: PAY ATTENTION, max pooling is used only in the encoder and when is set to true, the stride is set to 1
    'use_lstm_output': use_lstm_output
}

# Define the dataset
id_data = pd.read_csv('../data/id_df_final.csv')
seq_len = id_data.groupby('sequence').size()
seq_len = seq_len.to_dict()
seq_rain = id_data.groupby('sequence')['rain_category'].mean()
seq_rain = seq_rain.to_dict()
seq_df = pd.DataFrame({'seq_len': seq_len, 'seq_rain': seq_rain})
# Split the sequences in train and test set (80/20)
train_seq = seq_df.sample(frac=0.8, random_state=1)
test_seq = seq_df.drop(train_seq.index)
print(f"Average train sequence lenght: {train_seq['seq_len'].mean()}.")
print(f"Average test sequence lenght:, {test_seq['seq_len'].mean()}.")
print(f"Average test rain:, {train_seq['seq_rain'].mean()}.")
print(f"Average train rain:, {test_seq['seq_rain'].mean()}.")
print("")
# Get the sequences of the train and test set
train_seq_idx = train_seq.index
test_seq_idx = test_seq.index
train_data = id_data[id_data['sequence'].isin(train_seq_idx)]
test_data = id_data[id_data['sequence'].isin(test_seq_idx)]

id_data = None
seq_len = None
seq_rain = None
seq_df = None
train_seq = None
test_seq = None
train_seq_idx = None
test_seq_idx = None

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

# Loop over the dataset multiple times, with different sequence lengths to avoid the vanishing gradient problem
start_time = time.time()
max_seq_len = 10
for seq_len in range(2, max_seq_len):
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

    # Define a dataset of variable lenght
    train_dataset = SequenceDataset(train_data, '../../scratch/grey_tensor/', seq_len, seq_len)
    test_dataset = SequenceDataset(test_data, '../../scratch/grey_tensor/', seq_len, seq_len)
    dataloader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Number of elements to set to zero in the mask
    total_pixels = custom_model_config['in_shape'][1] * custom_model_config['in_shape'][2]
    num_zeros = int(total_pixels * ( seq_len / (max_seq_len * 2)))

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

        # Validation phase
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
        print("Elapsed time: {:.2f} minutes.".format((time.time() - start_time)/60))

print("")
print("Training complete!")
print("Total elapsed time: {:.2f} minutes.".format((time.time() - start_time)/60))

model_name = f"{out_folder}/model_{num_hidden[0]}_{num_hidden[1]}_{num_hidden[2]}_{num_hidden[3]}_{job_id}.pth"
th.save(model.state_dict(), model_name)
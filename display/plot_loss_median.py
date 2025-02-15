import re
import argparse
import matplotlib.pyplot as plt
import sys
import numpy as np

def compute_medians(filename):
    sequence_data = {}

    with open(filename, 'r') as file:
        current_seq = None
        for line in file:
            line = line.strip()

            if line.startswith("Training with sequence length"):
                parts = line.split()
                current_seq = int(parts[-1].strip('.'))
                if current_seq not in sequence_data:
                    sequence_data[current_seq] = []

            elif current_seq is not None:
                try:
                    loss_value = float(line)
                    sequence_data[current_seq].append(loss_value)
                except ValueError:
                    continue

    medians = {}
    for seq, values in sorted(sequence_data.items()):
        if values:
            median_value = np.median(values)
            medians[seq] = median_value
            print(f"Sequence Length {seq}: Median Loss = {median_value:.6f}")

    return medians

def parse_loss_file(filename):
    train_losses = []
    val_losses = []
    test_losses = []
    x_labels = []
    seq_lens = []

    with open(filename, 'r') as f:
        for line in f:
            train_match = re.search(r'Seq_Len: (\d+), Epoch \[(\d+)/(\d+)\] - Average Train Loss: ([0-9.]+)', line)
            val_match = re.search(r'Seq_Len: (\d+), Epoch \[(\d+)/(\d+)\] - Average Validation Loss: ([0-9.]+)', line)
            test_match = re.search(r'Seq_Len: (\d+), Epoch \[(\d+)/(\d+)\] - Average Test Loss: ([0-9.]+)', line)

            if train_match:
                seq_len, epoch, _, loss = train_match.groups()
                seq_len, epoch, loss = int(seq_len), int(epoch), float(loss)
                x_labels.append(f'(Epoch={epoch}, Seq_Len={seq_len})')
                train_losses.append(loss)
                seq_lens.append(seq_len)

            if val_match:
                _, _, _, loss = val_match.groups()
                val_losses.append(float(loss))

            if test_match:
                _, _, _, loss = test_match.groups()
                test_losses.append(float(loss))

    return train_losses, val_losses, test_losses, x_labels, seq_lens

def plot_losses(train_losses, val_losses, test_losses, x_labels, seq_lens, medians, out_file):
    plt.figure(figsize=(12, 6))

    plt.plot(x_labels, train_losses, label='Train Loss', linestyle='-', marker='o', color='blue')
    plt.plot(x_labels, val_losses, label='Validation Loss', linestyle='-.', marker='s', color='green')
    plt.plot(x_labels, test_losses, label='Test Loss', linestyle='--', marker='x', color='red')

    median_x = []
    median_y = []
    for i, seq in enumerate(seq_lens):
        if seq in medians:
            median_x.append(x_labels[i])
            median_y.append(medians[seq])

    plt.plot(median_x, median_y, label='Median Loss', linestyle=':', marker='d', color='purple')

    plt.xlabel('Epoch, Sequence Length')
    plt.ylabel('Loss')
    plt.title('Train, Validation, and Test Loss over Epochs and Sequence Lengths')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot train, validation, test losses and medians from log files.')
    parser.add_argument('--file', required=True, help='Input log file')
    parser.add_argument('--cleaned_file', required=True, help='Cleaned file for computing medians')
    parser.add_argument('--out_file', required=True, help='Output plot file')

    args = parser.parse_args()

    medians = compute_medians(args.cleaned_file)
    train_losses, val_losses, test_losses, x_labels, seq_lens = parse_loss_file(args.file)
    plot_losses(train_losses, val_losses, test_losses, x_labels, seq_lens, medians, args.out_file)

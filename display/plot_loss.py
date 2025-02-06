import re
import argparse
import matplotlib.pyplot as plt

def parse_loss_file(filename):
    train_losses = []
    test_losses = []
    val_losses = []
    x_labels = []
    
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
            
            if val_match:
                seq_len, epoch, _, loss = val_match.groups()
                seq_len, epoch, loss = int(seq_len), int(epoch), float(loss)
                val_losses.append(loss)
            
            if test_match:
                seq_len, epoch, _, loss = test_match.groups()
                seq_len, epoch, loss = int(seq_len), int(epoch), float(loss)
                test_losses.append(loss)
    
    return train_losses, val_losses, test_losses, x_labels

def plot_losses(train_losses, val_losses, test_losses, x_labels, out_file):
    plt.figure(figsize=(12, 6))
    
    plt.plot(x_labels, train_losses, label='Train Loss', linestyle='-', marker='o', color='blue')
    plt.plot(x_labels, val_losses, label='Validation Loss', linestyle='-.', marker='s', color='green')
    plt.plot(x_labels, test_losses, label='Test Loss', linestyle='--', marker='x', color='red')
    
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
    parser = argparse.ArgumentParser(description='Plot train, validation, and test losses from log file.')
    parser.add_argument('--file', required=True, help='Input log file')
    parser.add_argument('--out_file', required=True, help='Output plot file')
    
    args = parser.parse_args()
    train_losses, val_losses, test_losses, x_labels = parse_loss_file(args.file)
    plot_losses(train_losses, val_losses, test_losses, x_labels, args.out_file)



import re
import argparse
import matplotlib.pyplot as plt

def parse_loss_file(filename):
    train_losses = {}
    test_losses = {}
    seq_order = []
    
    with open(filename, 'r') as f:
        for line in f:
            train_match = re.search(r'Seq_Len: (\d+) Epoch \[(\d+)/(\d+)\] - Average Train Loss: ([0-9.]+)', line)
            test_match = re.search(r'Seq_Len: (\d+) Epoch \[(\d+)/(\d+)\] - Average Test Loss: ([0-9.]+)', line)
            train_match = re.search(r'Epoch \[(\d+)/(\d+)\] - Average Train Loss: ([0-9.]+)', line)
            test_match = re.search(r'Epoch \[(\d+)/(\d+)\] - Average Test Loss: ([0-9.]+)', line)

            if train_match:
                seq_len, epoch, _, loss = train_match.groups()
                seq_len, epoch = int(seq_len), int(epoch)
                loss = float(loss)
                if seq_len not in seq_order:
                    seq_order.append(seq_len)
                train_losses.setdefault(seq_len, []).append((epoch, loss))
            
            if test_match:
                seq_len, epoch, _, loss = test_match.groups()
                seq_len, epoch = int(seq_len), int(epoch)
                loss = float(loss)
                if seq_len not in seq_order:
                    seq_order.append(seq_len)
                test_losses.setdefault(seq_len, []).append((epoch, loss))
    print(train_losses)
    print(test_losses)
    
    return train_losses, test_losses, seq_order

def plot_losses(train_losses, test_losses, seq_order, out_file):
    plt.figure(figsize=(10, 6))
    
    for seq_len in seq_order:
        train_data = train_losses.get(seq_len, [])
        test_data = test_losses.get(seq_len, [])
        epochs, train_loss_values = zip(*train_data) if train_data else ([], [])
        _, test_loss_values = zip(*test_data) if test_data else ([], [])
        
        plt.plot(epochs, train_loss_values, label=f'Train Loss (Seq {seq_len})', linestyle='-', marker='o')
        plt.plot(epochs, test_loss_values, label=f'Test Loss (Seq {seq_len})', linestyle='--', marker='x')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(out_file)
    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot train and test losses from log file.')
    parser.add_argument('--file', required=True, help='Input log file')
    parser.add_argument('--out_file', required=True, help='Output plot file')
    
    args = parser.parse_args()
    train_losses, test_losses, seq_order = parse_loss_file(args.file)
    plot_losses(train_losses, test_losses, seq_order, args.out_file)

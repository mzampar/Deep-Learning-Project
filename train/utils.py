from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch as th
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn.functional as F

class SSIM_MSE_Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super(SSIM_MSE_Loss, self).__init__()
        self.alpha = alpha  # Weight for combining SSIM and MSE
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # Compute SSIM
        pred_channel = pred[:, :, 0, :, :]
        target_channel = target[:, :, 0, :, :]
        pred_channel = pred_channel.view(-1, 1, pred_channel.size(-2), pred_channel.size(-1))
        target_channel = target_channel.view(-1, 1, target_channel.size(-2), target_channel.size(-1))
        ssim_loss = 1 - ssim(pred_channel, target_channel, data_range=1.0)
        # Compute MSE
        mse_loss = self.mse_loss(pred, target)
        # Weighted combination
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss

class SequenceDataset(th.utils.data.Dataset):
    def __init__(self, id_df, tensor_dir, k_in=10, k_out=10):
        self.tensor_dir = tensor_dir
        self.k_in = k_in # Number of frames to be considered
        self.k_out = k_out
        self.id_seq_ds = pd.DataFrame([])

        data = []

        for i in id_df['sequence'].unique():
            seq_id = i
            len_seq = len(id_df[id_df['sequence'] == seq_id])
            frames = id_df[id_df['sequence'] == seq_id]
            
            for j in range(len_seq - k_in - k_out - 1):
                k_frames = frames.iloc[j:j+k_in][['id', 'rain_category']]
                id_frames_to_pred = frames.iloc[j+k_in:j+k_in+k_out][['id', 'rain_category']]
                k_frames = pd.concat([k_frames, id_frames_to_pred], ignore_index=True, axis=0)
                rain_category = np.mean(k_frames['rain_category'])
                ids = [int(id_) for id_ in k_frames['id'].tolist()]
                data.append(ids + [seq_id, rain_category])

        self.id_seq_ds = pd.DataFrame(data, columns=[f'id_{i}' for i in range(k_in + k_out)] + ['sequence', 'rain_category'])
       
        col_names = [f"input_{i}" for i in range(0, self.k_in)]
        col_names = col_names + [f"output_{i}" for i in range(0, self.k_out)]
        col_names = col_names + ["seq_id"] + ["rain_category"]

        self.id_seq_ds.columns = col_names

        self.id_seq_ds = self.id_seq_ds[self.id_seq_ds['rain_category'] > 3]
        self.id_seq_ds = self.id_seq_ds.astype(int)


    def __getitem__(self, index):
        # Get the row using the index
        row = self.id_seq_ds.iloc[index]
        # Get the sequence
        in_seq = row.iloc[:self.k_in]
        in_seq_tensor = th.stack([th.load(os.path.join(self.tensor_dir, f"tensor_{frame}.pt"), weights_only=True) for frame in in_seq])
        out_seq = row.iloc[self.k_in:self.k_in+self.k_out]
        out_seq_tensor = th.stack([th.load(os.path.join(self.tensor_dir, f"tensor_{frame}.pt"), weights_only=True) for frame in out_seq])

        return in_seq_tensor, out_seq_tensor

    def __len__(self):
        return len(self.id_seq_ds)

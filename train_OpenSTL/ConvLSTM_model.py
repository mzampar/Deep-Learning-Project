import torch
import torch.nn as nn

from ConvLSTM_module import ConvLSTMCell


class ConvLSTM_Model(nn.Module):
    r"""ConvLSTM Model

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    Code taken from https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/openstl/models/convlstm_model.py

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(ConvLSTM_Model, self).__init__()
        T, C, H, W = configs['in_shape']

        self.configs = configs
        self.frame_channel = configs['patch_size'] * configs['patch_size'] * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs['patch_size']
        width = W // configs['patch_size']
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, configs['filter_size'],
                                       configs['stride'], configs['layer_norm'])
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames, mask_true, **kwargs):
        # frames: [batch, length, channel, height, width]
        device = frames.device

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        # Initialize hidden and cell states
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

            # part of schedule sampling (commented at the end of file)

        # Loop through the entire sequence (pre_seq_length + aft_seq_length)
        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # Directly use the actual frame (no scheduled sampling)
            if t < self.configs.pre_seq_length:
                net = frames[:, t]  # Use input frame for pre-sequence
            else:
                net = frames[:, t]  # Use actual frame for aft-sequence as well

            # Apply ConvLSTM cell
            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            # Process through additional layers
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            # Generate the output frame
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1)

        return next_frames
    

        """
        for t in range(self.configs['pre_seq_length'] + self.configs['aft_seq_length'] - 1):
            # reverse schedule sampling
            if self.configs['reverse_scheduled_sampling'] == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs['pre_seq_length']:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs['pre_seq_length']] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs['pre_seq_length']]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

        """
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
                                       configs['stride'], configs['layer_norm'], configs['batch_size'])
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # frames_tensor: [batch, length, channel, height, width]
        device = frames_tensor.device

        batch = frames_tensor.shape[0]
        height = frames_tensor.shape[3]
        width = frames_tensor.shape[4]
        length = frames_tensor.shape[1]

        next_frames = []
        h_t_prev = []
        c_t_prev = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(device)
            h_t_prev.append(zeros)
            c_t_prev.append(zeros)

        # reverse schedule sampling: (todo)
        # here we manage the inputs: frames_tensor and mask_true

        for t in range(self.configs['pre_seq_length'] + self.configs['aft_seq_length'] + self.configs['target_seq_length']):
            # reverse schedule sampling
            if t < self.configs['pre_seq_length']:
                net = frames_tensor[:, t]
            elif t < self.configs['pre_seq_length'] + self.configs['aft_seq_length']:
                net = mask_true[:, t - self.configs['pre_seq_length']] * frames_tensor[:, t] + \
                        (1 - mask_true[:, t - self.configs['pre_seq_length']]) * x_gen
            else:
                net = x_gen

            h_t = []
            c_t = []

            a, b = self.cell_list[0](net, h_t_prev[0], c_t_prev[0])
            h_t.append(a)
            c_t.append(b)

            for i in range(1, self.num_layers):
                a, b = self.cell_list[i](h_t[i - 1], h_t_prev[i], c_t_prev[i])
                h_t.append(a)
                c_t.append(b)

            h_t_prev = h_t
            c_t_prev = c_t

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            if (t > self.configs['pre_seq_length'] + self.configs['aft_seq_length'] - 1):
                next_frames.append(x_gen)

        """
        if return_loss:
            loss = self.MSE_criterion(next_frames[self.configs[:'pre_seq_length']], frames_tensor)
        else:
            loss = None
        """

        return torch.stack(next_frames, dim=0)

    

        """
        for t in range(self.configs['pre_seq_length'] + self.configs['aft_seq_length'] - 1):
            # reverse schedule sampling
            if self.configs['reverse_scheduled_sampling'] == 1:
                if t == 0:
                    net = frames_tensor[:, t]
                else:
                    net = mask_true[:, t - 1] * frames_tensor[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs['pre_seq_length']:
                    net = frames_tensor[:, t]
                else:
                    net = mask_true[:, t - self.configs['pre_seq_length']] * frames_tensor[:, t] + \
                          (1 - mask_true[:, t - self.configs['pre_seq_length']]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

        """
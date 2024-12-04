import torch
import torch.nn as nn

from ConvLSTM_module import ConvLSTMCell


class ConvLSTM_Model(nn.Module):
    """ConvLSTM Model

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    Code readapted from https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/openstl/models/convlstm_model.py

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

        for i in range(num_layers):
        # vertical stack of ConvLSTM cells
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, configs['filter_size'],
                                       configs['stride'], configs['layer_norm'])
            )
        self.cell_list = nn.ModuleList(cell_list)
        # the last layer has to output the frame_channel
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, ):
        """
        We are probably following a different approach from the paper.
        We are processing one frame at a time, passing it vertically trough the layers, to get an output frame.
        During prediction, this output frame is then used as input for the next frame.
        During trianing we could use sheduled sampling to make the model able to use its own predictions also and not only the inputs when analysing the given sequence.
        When processing vertically a frame, we keep track of the hidden and cell states of each layer, that will be used when processing the next frame.
        I think this is more efficient because we only have to keep track of the hidden and cell states of the last (in time) frame processed.
        """

        # frames_tensor: [batch, length, channel, height, width]
        # mask_true: [batch, length, channel, height, width]
        device = frames_tensor.device
        batch = frames_tensor.shape[0]
        length = frames_tensor.shape[1]
        height = frames_tensor.shape[3]
        width = frames_tensor.shape[4]

        next_frames = []
        h_t_prev = []
        c_t_prev = []

        for i in range(self.num_layers):
            # the hidden and cell states of each layer for the first frame are initialized with zeros
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(device)
            h_t_prev.append(zeros)
            c_t_prev.append(zeros)

        # schedule sampling:
        # here we manage the inputs: frames_tensor and mask_true
        # if t < pre_seq_length, we use the true frames; 
        # if t < pre_seq_length + aft_seq_length, we mix the true frames and the generated frames; 
        # if t >= pre_seq_length + aft_seq_length, we use the generated frames

        x_gen = frames_tensor[:, 0]

        for t in range(2*length):
            if t < length//2:
                net = frames_tensor[:, t]
            elif t < length:
                net = mask_true[:, t] * frames_tensor[:, t] + (1 - mask_true[:, t]) * x_gen
            else:
                net = x_gen
            # keeping track of the hidden and cell states of each layer
            h_t = []
            c_t = []
            a, b = self.cell_list[0](net, h_t_prev[0], c_t_prev[0])
            h_t.append(a)
            c_t.append(b)
            for i in range(1, self.num_layers):
                a, b = self.cell_list[i](h_t[i - 1], h_t_prev[i], c_t_prev[i])
                h_t.append(a)
                c_t.append(b)
            # update the hidden and cell states of each layer
            h_t_prev = h_t
            c_t_prev = c_t
            # the last layer generates the output frame
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            # -2 because we start from 0 and the k_in-th output frame is the first generated frame
            if (t > length - 2):
                # keep track of the generated frames
                next_frames.append(x_gen)

        # we could also return the loss btw the predicted frames and the true frames
        # we discard the last frame beacuse it would be the k_out+1-th frame
        return torch.stack(next_frames[:-1], dim=1)
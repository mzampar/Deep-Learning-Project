import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm, transpose=False, bias=False):
        super(ConvLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        if (filter_size - stride) % 2 == 0:
            self.trans_padding = (filter_size - stride)//2
            self.out_padding = 0
        else:
            self.trans_padding = (filter_size - stride)//2 + 1
            self.out_padding = 1
        self.padding = (filter_size - 1) // 2
        # Initialize weights for the context
        self.context_input = nn.Parameter(torch.empty(1, num_hidden, height, width))
        self.context_output = nn.Parameter(torch.empty(1, num_hidden, height, width))
        self.context_forget = nn.Parameter(torch.empty(1, num_hidden, height, width))

        # Apply Xavier initialization to context parameters
        nn.init.xavier_uniform_(self.context_input)
        nn.init.xavier_uniform_(self.context_output)
        nn.init.xavier_uniform_(self.context_forget)
        # We could also add the bias

        # Convolutions for the input and hidden states
        # Height after transpose: output_height = (height−1)*stride − 2*padding + kernel_size + output_padding

        if layer_norm:
            if transpose:
                # num_hidden * 4 because we have 4 gates
                self.conv_x = nn.Sequential(
                    nn.ConvTranspose2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                        stride=stride, padding=self.trans_padding, bias=bias, output_padding=self.out_padding),
                    nn.LayerNorm([num_hidden * 4, height, width])
                )
                self.conv_h = nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                        stride=1, padding=self.padding, bias=bias),
                    nn.LayerNorm([num_hidden * 4, height, width])
                )
                # conv_o is not used in the forward pass
                """
                self.conv_o = nn.Sequential(
                    nn.ConvTranspose2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                        stride=stride, padding=self.trans_padding, bias=False, output_padding=self.out_padding),
                    nn.LayerNorm([num_hidden * 2, height, width])
                )
                """
            else:
                self.conv_x = nn.Sequential(
                    nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=bias),
                    nn.LayerNorm([num_hidden * 4, height, width])
                )
                self.conv_h = nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                        stride=1, padding=self.padding, bias=bias),
                    nn.LayerNorm([num_hidden * 4, height, width])
                )
                # conv_o is not used in the forward pass
                """
                self.conv_o = nn.Sequential(
                    nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
                    nn.LayerNorm([num_hidden * 2, height, width])
                )
                """
        else:
            if transpose:
                self.conv_x = nn.ConvTranspose2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                                stride=stride, padding=self.trans_padding, bias=bias, output_padding=self.out_padding)
                self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                                stride=1, padding=self.padding, bias=bias)
                # conv_o is not used in the forward pass
                """
                self.conv_o = nn.ConvTranspose2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                                stride=stride, padding=self.trans_padding, bias=False, output_padding=self.out_padding)
                """
            else:
                self.conv_x = nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                                stride=stride, padding=self.padding, bias=bias)
                self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                                stride=1, padding=self.padding, bias=bias)
                # conv_o is not used in the forward pass
                """
                self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                                stride=stride, padding=self.padding, bias=False)
                """


    def forward(self, x_t_new, h_t, c_t):
        x_concat = self.conv_x(x_t_new)
        h_concat = self.conv_h(h_t)

        i_x, f_x, c_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, c_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        # computing the four gates as in the paper
        i_t = torch.sigmoid(i_x + i_h + self.context_input * c_t) 
        f_t = torch.sigmoid(f_x + f_h + self.context_forget * c_t)

        c_new = f_t * c_t + i_t * torch.tanh(c_x + c_h)
        o_t = torch.sigmoid(o_x + o_h + self.context_output * c_new)
        h_new = o_t * torch.tanh(c_new)

        return h_new, c_new
import torch
import torch.nn as nn


def create_conv_block(in_channels, out_channels, kernel_size, stride, padding, bias, 
                      transpose=False, leaky_slope=None, layer_norm=False, height=None, width=None, 
                      trans_padding=None, out_padding=None):
    """
    Helper function to create convolutional blocks with optional LayerNorm or LeakyReLU.
    """
    layers = []
    
    if transpose:
        conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=trans_padding, bias=bias, output_padding=out_padding
        )
    else:
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias
        )
    
    layers.append(conv)
    
    if layer_norm and height and width:
        layers.append(nn.LayerNorm([out_channels, height, width]))
    
    if leaky_slope is not None:
        layers.append(nn.LeakyReLU(leaky_slope))
    
    return nn.Sequential(*layers)

class ConvLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm, transpose=False, bias=False, leaky_slope=None):
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

        # Main constructor logic
        if transpose:
            self.conv_x = create_conv_block(
                in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride,
                padding=self.trans_padding, bias=bias, transpose=True, 
                leaky_slope=leaky_slope, layer_norm=layer_norm, height=height, width=width,
                trans_padding=self.trans_padding, out_padding=self.out_padding
            )
            self.conv_h = create_conv_block(
                num_hidden, num_hidden * 4, kernel_size=filter_size, stride=1, 
                padding=self.padding, bias=bias, transpose=False, 
                leaky_slope=leaky_slope, layer_norm=layer_norm, height=height, width=width
            )
        else:
            self.conv_x = create_conv_block(
                in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride,
                padding=self.padding, bias=bias, transpose=False, 
                leaky_slope=leaky_slope, layer_norm=layer_norm, height=height, width=width
            )
            self.conv_h = create_conv_block(
                num_hidden, num_hidden * 4, kernel_size=filter_size, stride=1, 
                padding=self.padding, bias=bias, transpose=False, 
                leaky_slope=leaky_slope, layer_norm=layer_norm, height=height, width=width
            )



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
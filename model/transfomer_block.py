import torch as t
import numpy as np
# from module import *
# from utils import get_positional_table, get_sinusoid_encoding_table
import copy
import torch.nn as nn
# from utils.text.symbols import symbols
import math
import config.hparams as hp


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class FFN(nn.Module):
    """
    Positionwise Feed-Forward Network
    """

    def __init__(self, num_hidden):
        """
        :param num_hidden: dimension of hidden
        """
        super(FFN, self).__init__()
        self.w_1 = Conv(num_hidden, num_hidden * 4, kernel_size=1, w_init='relu')
        self.w_2 = Conv(num_hidden * 4, num_hidden, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, input_):
        # FFN Network
        x = input_.transpose(1, 2)
        x = self.w_2(t.relu(self.w_1(x)))
        x = x.transpose(1, 2)

        # residual connection
        x = x + input_

        # dropout
        x = self.dropout(x)

        # layer normalization
        x = self.layer_norm(x)

        return x


class Conv(nn.Module):
    """
    Convolution Module
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x


class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class EncoderPrenet(nn.Module):
    """
    Pre-network for Encoder consists of convolution networks.
    """
    def __init__(self, embedding_size, channels):
        super(EncoderPrenet, self).__init__()
        self.embedding_size = embedding_size

        self.conv1d_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        in_channels, out_channels = embedding_size, channels
        kernel_size = hp.enc_conv1d_kernel_size
        for i in range(hp.enc_conv1d_layers):
            conv1d = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          padding=int(np.floor(kernel_size / 2)), w_init='relu')
            self.conv1d_layers.append(conv1d)

            batch_norm = nn.BatchNorm1d(out_channels)
            self.bn_layers.append(batch_norm)

            dropout_layer = nn.Dropout(p=hp.enc_conv1d_dropout_rate)
            self.dropout_layers.append(dropout_layer)

            in_channels, out_channels = out_channels, out_channels

        self.projection = Linear(out_channels, channels)

    def forward(self, input):
        """
        :param input: B*T*d
        :return:
        """
        input = input.transpose(1, 2)       # B*d*T

        for conv1d, bn, dropout in zip(self.conv1d_layers, self.bn_layers, self.dropout_layers):
            input = dropout(t.relu(bn(conv1d(input))))     # B*d*T

        input = input.transpose(1, 2)       # B*T*d
        input = self.projection(input)

        return input



class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """
    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)
        self.attention = None

    def forward(self, key, value, query, mask=None, query_mask=None):
        # Get attention score
        attn = t.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        # Masking to ignore padding (key side)
        if mask is not None:

            attn = attn.masked_fill(mask, -2 ** 32 + 1)
            attn = t.softmax(attn, dim=-1)
        else:
            attn = t.softmax(attn, dim=-1)

        # Masking to ignore padding (query side)   # 只是这里是q=k=v
        if query_mask is not None:
            attn = attn * query_mask  # 128,804,64

        # Dropout
        # attn = self.attn_dropout(attn)
        self.attention = self.attn_dropout(attn).view(key.size(0)//4, 4, -1, key.size(1))

        # Get Context Vector
        result = t.bmm(attn, value)

        return result, self.attention  # 128,804,158



class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=hp.self_att_block_res_dropout)
        self.final_linear = Linear(num_hidden * 2, num_hidden)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, memory, decoder_input, mask=None, query_mask=None):

        batch_size = memory.size(0)  # 这里memory == decoder_input
        seq_k = memory.size(1)
        seq_q = decoder_input.size(1)

        # Repeat masks h times
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k)
            query_mask = query_mask.repeat(self.h, 1, 1)  # （32，169，169）————>(128,169,169)
        if mask is not None:
            mask = mask.repeat(self.h, 1, 1)  # （32，169，169）————>(128,169,169)

        # Make multihead
        key = self.key(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(memory).view(batch_size, seq_k, self.h,
                                        self.num_hidden_per_attn)  # (32,164,256)-->()32,164,4,64
        query = self.query(decoder_input).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query, mask=mask, query_mask=query_mask)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = t.cat([decoder_input, result], dim=-1)  # 把input256+result256 cat 在一起..成512

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + decoder_input

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns



def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):  # 0-255
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):  # 0-1023
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]  # 循环256 ，256列

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)]) # 1024，256  循环1-1024  1024次

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i  每一行的偶数列
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1  对每一行的奇数列

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return t.FloatTensor(sinusoid_table)







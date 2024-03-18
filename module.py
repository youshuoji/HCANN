import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from einops import rearrange

## transformer
class Position_embedding_T(nn.Module):
    def __init__(self, channels, dropout, sample_time=1024*1):
        super(Position_embedding_T, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position_time = torch.rand(size=(channels, sample_time))
        self.register_buffer('position_time', position_time)

    def forward(self, x):
        x = x + Variable(nn.ReLU(self.position_time), requires_grad=True)
        return self.dropout(x)

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, heads, dim_head, dropout=0.):
        super(Multi_Head_Attention, self).__init__()
        self.heads = heads
        # assert dim_model % heads == 0
        inner_dim = heads * dim_head

        self.to_qkv = nn.Linear(dim_model, inner_dim * 3, bias=False)
        # self.layer_norm = nn.LayerNorm(dim_model)
        self.attend = nn.Softmax(dim=-1)
        self.scale = dim_head ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attd = self.attend(dots)
        out = torch.matmul(attd, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, channel, dropout=0.0):
        super().__init__()

        self.feed_net = nn.Sequential(
            nn.ZeroPad2d((0, 0, 2, 2)),
            nn.Conv2d(in_channels=channel, out_channels=channel*hidden, kernel_size=(5, 1), stride=1, bias=False),
            nn.BatchNorm2d(channel*hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=channel*hidden, out_channels=channel, kernel_size=1, stride=1, bias=False),

            nn.BatchNorm2d(channel),
            nn.ELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.feed_net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.):
        super().__init__()
        self.heads = heads
        assert dim % heads == 0
        self.dim_head = dim // self.heads

        self.scale = self.dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, heads * self.dim_head * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(heads * self.dim_head, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class transformer(nn.Module):
    def __init__(self, depth, dim, heads, hidden_feed, channel, dropout=0.):
        super(transformer, self).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout)),
                Position_wise_Feed_Forward(dim, hidden_feed, channel, dropout)
            ]))

    def forward(self, x):
        for attn, feed in self.layers:
            x = attn(x) + x
            # x = attn(x)
            x = feed(x) + x
        return x

class temporal_block(nn.Module):
    def __init__(self, input_dim, output_dim, kernek_size, pad_len, stride=1, expansion=2, dropout=0.):
        super().__init__()

        self.stride = stride

        hidden_dim = int(input_dim * expansion)
        self.use_res_connect = self.stride == 1 and input_dim == output_dim

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.ZeroPad2d(pad_len),
                nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernek_size, stride=stride,
                          groups=hidden_dim, bias=True),
                nn.BatchNorm2d(hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Conv2d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm2d(output_dim)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm2d(hidden_dim),
                nn.ELU(),

                nn.ZeroPad2d(pad_len),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernek_size, stride=stride,
                          groups=hidden_dim, bias=True),
                nn.BatchNorm2d(hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),

                nn.Conv2d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm2d(output_dim)
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

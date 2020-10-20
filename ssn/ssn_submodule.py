import sys
sys.path.append("..")

import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils.net_utils import compute_differentiable_params
from params import params

parameter = params().get_params()
def get_layer_info(out_channels):
    if parameter.norm == 'batch_norm':
        norm_layer = nn.BatchNorm2d(out_channels, momentum=0.9)
    elif parameter.norm == 'group_norm':
        if out_channels >= 32:
            group_num = 32
            if out_channels % group_num != 0:
                group_num = 16
        else:
            group_num = 1
            
        norm_layer = nn.GroupNorm(group_num, out_channels)
    elif parameter.norm == 'None':
        norm_layer = None
    else:
        raise Exception('norm name error')
    
    if parameter.prelu:
        activation_func = nn.PReLU(out_channels)
    else:
        activation_func = nn.ReLU()
        
    return norm_layer, activation_func

class Conv(nn.Module):
    """ (convolution => [BN] => ReLU) """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, conv_stride=1, padding=1, bias=True, force_relu=False):
        super().__init__()
        
        norm_layer, activation_func = get_layer_info(out_channels)
        if force_relu:
            activation_func = nn.ReLU()
            
        if norm_layer is not None:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,stride=conv_stride, kernel_size=kernel_size, padding=padding, bias=bias),
                norm_layer,
                activation_func)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,stride=conv_stride, kernel_size=kernel_size, padding=padding, bias=bias),
                activation_func)

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """ Upscaling then conv """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        parameter = params().get_params()
        if parameter.prelu:
            activation_func = 'prelu'
        else:
            activation_func = 'relu'
        norm_layer, activation_func = get_layer_info(out_channels)
        
        up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.Sequential(
                    up_layer,
                    Conv(in_channels, in_channels//4),
                    norm_layer,
                    activation_func)

    def forward(self, x):
        return self.up(x)

class Up_Stream(nn.Module):
    """ Up Stream Sequence """
    
    def __init__(self, out_channels=3):
        super(Up_Stream, self).__init__()
        input_channel = 512

        self.up_16_16_1 = Conv(input_channel, 256)
        self.up_16_16_2 = Conv(768, 512)
        self.up_16_16_3 = Conv(1024, 512)
        self.up_16_32 = Up(1024, 256)

        self.up_32_32_1 = Conv(512, 256)

        self.up_32_64 = Up(512, 128)
        self.up_64_64_1 = Conv(256, 128)

        self.up_64_128 = Up(256, 64)
        self.up_128_128_1 = Conv(128, 64)

        self.up_128_256 = Up(128, 32)
        self.out_conv = Conv(64, out_channels, force_relu=True)
        
    def forward(self, l, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
        # batch_size, c, h, w = l.size()
        
        # import pdb; pdb.set_trace()
        # multiple channel ibl
        y = l.view(-1, 512, 1, 1).repeat(1, 1, 16, 16)

        y = self.up_16_16_1(y)    # 256 x 16 x 16

        y = torch.cat((x10,y), dim=1)   # 768 x 16 x 16
        # print(y.size())

        y = self.up_16_16_2(y)          # 512 x 16 x 16
        # print(y.size())

        y= torch.cat((x9,y), dim=1)     # 1024 x 16 x 16
        # print(y.size())
        
        # import pdb; pdb.set_trace()
        y = self.up_16_16_3(y)          # 512 x 16 x 16
        # print(y.size())

        y = torch.cat((x8, y), dim=1)   # 1024 x 16 x 16
        # print(y.size())
        
        # import pdb; pdb.set_trace()
        y = self.up_16_32(y)            # 256 x 32 x 32
        # print(y.size())

        
        y = torch.cat((x7, y), dim=1)
        y = self.up_32_32_1(y)          # 256 x 32 x 32
        # print(y.size())

        y = torch.cat((x6, y), dim=1)
        y = self.up_32_64(y)
        # print(y.size())
        y = torch.cat((x5, y), dim=1)
        y = self.up_64_64_1(y)          # 128 x 64 x 64
        # print(y.size())

        y = torch.cat((x4, y), dim=1)
        y= self.up_64_128(y)
        # print(y.size())
        y = torch.cat((x3, y), dim=1)
        y = self.up_128_128_1(y)        # 64 x 128 x 128
        # print(y.size())

        y = torch.cat((x2, y), dim=1)
        y = self.up_128_256(y)          # 32 x 256 x 256
        # print(y.size())
        
        y = torch.cat((x1, y), dim=1)

        y = self.out_conv(y)          # 3 x 256 x 256
        
        return y
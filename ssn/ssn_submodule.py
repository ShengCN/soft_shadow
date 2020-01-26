import sys
sys.path.append("..")

import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils.net_utils import compute_differentiable_params
from params import params

class Conv(nn.Module):
    """ (convolution => [BN] => ReLU) """
    
    def __init__(self, in_channels, out_channels, conv_stride, activation_func='relu'):
        super().__init__()

        if activation_func == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,stride=conv_stride, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, momentum=0.9),
                nn.ReLU()
            )
        elif activation_func == 'sigmoid':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,stride=conv_stride, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, momentum=0.9),
                nn.Sigmoid()
            )
        elif activation_func == 'prelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,stride=conv_stride, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels,momentum=0.9),
                nn.PReLU()
            )
        elif activation_func =='softplus':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,stride=conv_stride, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels,momentum=0.9),
                nn.Softplus()
            )
        else:
            assert False


    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """ Upscaling then conv """
    
    def __init__(self, in_channels, out_channels, activation_func='relu'):
        super().__init__()
        parameter = params()
        kernel_size = parameter.get_trans_conv_kernel()
        if activation_func == 'prelu':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2),
                nn.BatchNorm2d(out_channels,momentum=0.9),
                nn.PReLU()
            )
        elif activation_func == 'relu':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2),
                nn.BatchNorm2d(out_channels,momentum=0.9),
                nn.ReLU()
            )

    def forward(self, x):
        return self.up(x)

class Up_Stream(nn.Module):
    """ Up Stream Sequence """
    
    def __init__(self, out_channels=3):
        super(Up_Stream, self).__init__()
        self.up_16_16_1 = Conv(512, 256, 1)
        self.up_16_16_2 = Conv(768, 512, 1)
        self.up_16_16_3 = Conv(1024, 512, 1)

        self.up_16_32 = Up(1024, 256)
        self.up_32_32_1 = Conv(512, 256, 1)

        self.up_32_64 = Up(512, 128)
        self.up_64_64_1 = Conv(256, 128, 1)

        self.up_64_128 = Up(256, 64)
        self.up_128_128_1 = Conv(128, 64, 1)

        self.up_128_256 = Up(128, 32)
        self.out_conv = Conv(64, out_channels, 1, activation_func='sigmoid')
        
    def forward(self, l, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
        # tiled_l = torch.cat((l.view(-1, 6, 16, 16).repeat(1, 512 // 6, 1, 1), l.view(-1, 6, 16, 16)[:, 0:2, :, :]), dim=1)
        batch_size, c, h, w = l.size()
        tiled_l = l.view(-1, 512, 1, 1).repeat(1, 1, 16, 16)
        # print("tiled_l: {}".format(tiled_l.size()))

        y = self.up_16_16_1(tiled_l)    # 256 x 16 x 16
        # print(y.size())
        
        # import pdb; pdb.set_trace()
        
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
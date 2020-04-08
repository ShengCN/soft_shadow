import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssn_submodule import Conv, Up, Up_Stream, get_layer_info
from params import params

class Relight_SSN(nn.Module):
    """ Implementation of Relighting Net """

    def __init__(self, n_channels=3, out_channels=3):
        super(Relight_SSN, self).__init__()
        
        # import pdb; pdb.set_trace()
        parameter = params().get_params()
        if parameter.prelu:
            activation_func = 'prelu'
        else:
            activation_func = 'relu'
            
        norm_layer, activation_func = get_layer_info(32 - n_channels, activation_func)

        # norm_layer, activation_func = get_layer_info(32 - n_channels)
        # norm_layer = nn.BatchNorm2d(32 - n_channels, momentum=0.9)
        self.in_conv = nn.Sequential(
            nn.Conv2d(n_channels, 32 - n_channels, kernel_size=7, padding=3, bias=True),
            norm_layer,
            activation_func
        )
        
        self.down_256_128  = Conv(32, 64, 2)
        self.down_128_128  = Conv(64, 64, 1)
        self.down_128_64   = Conv(64, 128, 2)
        self.down_64_64    = Conv(128, 128, 1)
        self.down_64_32    = Conv(128, 256, 2)
        self.down_32_32    = Conv(256, 256, 1)
        self.down_32_16    = Conv(256, 512, 2)
        self.down_16_16_1  = Conv(512, 512, 1)
        self.down_16_16_2  = Conv(512, 512, 1)
        self.down_16_16_3  = Conv(512, 512, 1)
        self.to_bottleneck = Conv(512, 2, 1)
        
        self.up_stream = Up_Stream(out_channels)

    """
        Input is (source image, target light, source light, )
        Output is: predicted new image, predicted source light, self-supervision image
    """
    def forward(self, x, tl):
        x1 = self.in_conv(x)  # 29 x 256 x 256
        # print(x1.size())
        # import pdb; pdb.set_trace()
        
        x1 = torch.cat((x, x1), dim=1)  # 32 x 256 x 256 todo_check the dim parameters
        # print(x1.size())

        x2 = self.down_256_128(x1)  # 64 x 128 x 128
        # print(x2.size())

        x3 = self.down_128_128(x2)  # 64 x 128 x 128
        # print(x3.size())

        x4 = self.down_128_64(x3)  # 128 x 64 x 64
        # print(x4.size())

        x5 = self.down_64_64(x4)  # 128 x 64 x 64
        # print(x5.size())

        x6 = self.down_64_32(x5)  # 256 x 32 x 32
        # print(x6.size())

        x7 = self.down_32_32(x6)  # 256 x 32 x 32
        # print(x7.size())

        x8 = self.down_32_16(x7)  # 512 x 16 x 16
        # print(x8.size())

        x9 = self.down_16_16_1(x8)  # 512 x 16 x 16
        # print(x9.size())

        x10 = self.down_16_16_2(x9)  # 512 x 16 x 16
        # print(x10.size())

        x11 = self.down_16_16_3(x10)  # 512 x 16 x 16
        # print(x11.size())

        out_light = self.to_bottleneck(x11)  # 6 x 16 x 16

        ty = self.up_stream(tl, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11)
        # sy = self.up_stream(out_light, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11)

        return ty, out_light
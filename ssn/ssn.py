import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssn_submodule import Conv, Up, Up_Stream, get_layer_info, add_coords
from params import params

class Relight_SSN(nn.Module):
    """ Implementation of Relighting Net """

    def __init__(self, n_channels=3, out_channels=3):
        super(Relight_SSN, self).__init__()
        
        parameter = params().get_params()
        if parameter.prelu:
            activation_func = 'prelu'
        else:
            activation_func = 'relu'
        
        norm_layer, activation_func = get_layer_info(32 - n_channels, activation_func)

        if parameter.baseline and (norm_layer is not None):
            self.in_conv = nn.Sequential(
                nn.Conv2d(n_channels, 32 - n_channels, kernel_size=7, padding=3, bias=True),
                norm_layer,
                activation_func
            )
        elif norm_layer is None:
            self.in_conv = nn.Sequential(
                nn.Conv2d(n_channels, 32 - n_channels, kernel_size=7, padding=3, bias=True),
                activation_func
            )

        if parameter.coordconv:
            self.in_conv       = Conv(n_channels, 32 - n_channels, kernel_size=7, conv_stride=1, padding=3, bias=True)

        self.down_256_128  = Conv(32, 64, conv_stride=2)
        self.down_128_128  = Conv(64, 64, conv_stride=1)
        self.down_128_64   = Conv(64, 128, conv_stride=2)
        self.down_64_64    = Conv(128, 128, conv_stride=1)
        self.down_64_32    = Conv(128, 256, conv_stride=2)
        self.down_32_32    = Conv(256, 256, conv_stride=1)
        self.down_32_16    = Conv(256, 512, conv_stride=2)
        self.down_16_16_1  = Conv(512, 512, conv_stride=1)
        self.down_16_16_2  = Conv(512, 512, conv_stride=1)
        self.down_16_16_3  = Conv(512, 512, conv_stride=1)
        self.to_bottleneck = Conv(512, 2, conv_stride=1)
        
        self.up_stream = Up_Stream(out_channels)

    """
        Input is (source image, target light, source light, )
        Output is: predicted new image, predicted source light, self-supervision image
    """
    def forward(self, x, tl):

        x1 = self.in_conv(x)  # 29 x 256 x 256

        x1 = torch.cat((x, x1), dim=1)  # 32 x 256 x 256 

        x2 = self.down_256_128(x1)  # 64 x 128 x 128

        x3 = self.down_128_128(x2)  # 64 x 128 x 128

        x4 = self.down_128_64(x3)  # 128 x 64 x 64

        x5 = self.down_64_64(x4)  # 128 x 64 x 64

        x6 = self.down_64_32(x5)  # 256 x 32 x 32

        x7 = self.down_32_32(x6)  # 256 x 32 x 32

        x8 = self.down_32_16(x7)  # 512 x 16 x 16

        x9 = self.down_16_16_1(x8)  # 512 x 16 x 16

        x10 = self.down_16_16_2(x9)  # 512 x 16 x 16

        x11 = self.down_16_16_3(x10)  # 512 x 16 x 16

        out_light = self.to_bottleneck(x11)  # 6 x 16 x 16

        ty = self.up_stream(tl, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11)
        # sy = self.up_stream(out_light, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11)

        return ty, out_light